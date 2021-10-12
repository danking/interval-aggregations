from typing import List, Tuple, Optional
import hail as hl
import hailtop.batch as hb
import functools
import asyncio
import json
import argparse

from hailtop.utils import grouped, secret_alnum_string, bounded_gather
from hailtop.aiocloud.aiogoogle import GoogleStorageAsyncFS
from hailtop.aiotools.fs import AsyncFS


def aggregate_by_intervals(index, intervals, uuid) -> List[str]:
    import hail as hl

    hl.init(master='local[1]')

    exomes_vds = hl.vds.read_vds('gs://ccdg/vds/split_200k_ccdg_exome.vds')
    exomes_ref_mt = exomes_vds.reference_data
    exomes_ref_mt = exomes_ref_mt.annotate_entries(base_cnt = exomes_ref_mt.END - exomes_ref_mt.locus.position)

    paths = [
        aggregate_interval_to_file(exomes_ref_mt, interval, index, uuid)
        for interval in intervals]
    return paths


def aggregate_interval_to_file(exomes_ref_mt, the_interval, index: int, uuid: str) -> str:
    import hail as hl
    import sys
    from hailtop.utils import secret_alnum_string

    attempt_id = secret_alnum_string(5)
    path = f'gs://ccdg-30day-temp/dking/{uuid}/output_{index}_{attempt_id}.tsv'
    print(f'writing to {path}')
    sys.stdout.flush()
    sys.stderr.flush()
    the_interval = hl.literal(the_interval)
    exomes_ref_mt = exomes_ref_mt.filter_rows(the_interval.contains(exomes_ref_mt.locus))
    exomes_ref_mt.group_rows_by(
        interval=the_interval
    ).aggregate(
        n_bases = hl.agg.sum(exomes_ref_mt.base_cnt)
    ).key_rows_by(
        interval=the_interval
    ).n_bases.export(
        path
    )
    return path


def create_final_mt(combined_tsv: str, final_mt_path: str, partitions: int):
    import hail as hl
    mt = hl.import_matrix_table(combined_tsv, row_fields={'interval': hl.tstr})
    mt = mt.repartition(partitions)
    mt = mt.key_rows_by(interval=hl.parse_locus_interval(mt.interval, reference_genome='GRCh38'))
    mt.write(final_mt_path)


async def construct_aggregation_job(group, cpus, uuid: str, b: hb.Batch, fs: AsyncFS) -> str:
    index = group.group_index
    intervals = group.intervals
    result_path = f'gs://ccdg-30day-temp/dking/{uuid}/successful_paths_{index}.json'

    if not await fs.exists(result_path):
        j = b.new_python_job()
        j.cpu(cpus)
        j.env("PYSPARK_SUBMIT_ARGS", "--driver-memory 3g --executor-memory 3g pyspark-shell")
        result = j.call(aggregate_by_intervals, index, intervals, uuid)
        b.write_output(result.as_json(), result_path)

    return result_path


async def construct_aggregation_batch(grouped_intervals, cpus, uuid: str) -> Tuple[hb.Batch, List[str]]:
    b = hb.Batch(
        default_python_image='hailgenetics/hail:0.2.77',
        backend=hb.ServiceBackend(billing_project='hail'),
        name=f'{uuid}: interval statistics'
    )

    async with GoogleStorageAsyncFS() as fs:
        result_paths = await asyncio.gather(*[
            construct_aggregation_job(group, cpus, uuid, b, fs)
            for group in grouped_intervals])

    return b, result_paths


async def async_main(rerun: bool = False, prefix_rows: Optional[int] = None, group_size: Optional[int] = None):
    if hl.hadoop_exists('grouped-intervals.t') and not rerun:
        print(' >> skipping grouped intervals creation')
    else:
        print(' >> running grouped intervals creation')
        uuid = secret_alnum_string(5)
        intervals = hl.import_locus_intervals(
            "gs://gcp-public-data--broad-references/hg38/v0/exome_calling_regions.v1.interval_list", reference_genome='GRCh38')
        if prefix_rows is not None:
            intervals = intervals.head(prefix_rows)
        if group_size is None:
            group_size = 20
        intervals = intervals.annotate_globals(
            uuid=uuid,
            prefix_rows=prefix_rows,
            group_size=group_size
        )
        intervals = intervals.add_index('idx')
        intervals = intervals.group_by(
            group_index = intervals.idx // group_size,
        ).aggregate(
            intervals = hl.agg.collect(intervals.interval)
        )
        intervals.write('grouped-intervals.t', overwrite=True)

    grouped_intervals_table = hl.read_table('grouped-intervals.t')
    grouped_intervals = grouped_intervals_table.collect()
    uuid = grouped_intervals_table.uuid.collect()[0]
    if prefix_rows:
        assert prefix_rows == grouped_intervals_table.prefix_rows.collect()[0]
    if group_size:
        assert group_size == grouped_intervals_table.group_size.collect()[0]

    cpus = 2  # must be at least 2, for some large intervals 4, 8, or 16 may be necessary

    print(f'uuid: {uuid}; cpus: {cpus}; location: gs://ccdg-30day-temp/dking/{uuid}')

    b, result_paths = await construct_aggregation_batch(grouped_intervals, cpus, uuid)
    if len(b._jobs) == 0:
        print(' >> skipping aggregations')
    else:
        print(' >> running aggregations')
        batch_handle = b.run()
        batch_status = batch_handle.status()

        if batch_status['state'] != 'success':
            print(' >> first batch failed, halting.')
            return

    combined_tsv = f'gs://ccdg-30day-temp/dking/{uuid}/total-final-result.tsv'

    if hl.hadoop_exists(combined_tsv):
        print(' >> skipping TSV combination')
    else:
        print(' >> running TSV combination')
        async with GoogleStorageAsyncFS() as fs:
            async def read_as_json(path):
                return json.loads(await fs.read(path))

            results = await bounded_gather(*[
                functools.partial(read_as_json, path) for path in result_paths
            ], parallelism=50)

        mt_paths = [path
                    for paths in results
                    for path in paths]

        b = hb.Batch(
            backend=hb.ServiceBackend(billing_project='hail'),
            name=f'{uuid}: combine tsvs'
        )

        def combine_tsvs_with_headers(b, name, xs):
            j = b.new_job(name=name)
            j.command(f'head -n 1 {xs[0]} > {j.ofile}')
            j.command(f'tail -n +2 -q {" ".join(xs)} >> {j.ofile}')
            return j.ofile

        combined = hb.utils._combine(
            combine_tsvs_with_headers,
            b,
            'combine-tsvs-with-headers',
            [b.read_input(path) for path in mt_paths])

        b.write_output(combined, combined_tsv)

        batch_handle = b.run()
        batch_status = batch_handle.status()

        if batch_status['state'] != 'success':
            print(' >> combine batch failed, halting.')
            return

    final_mt_path = f'gs://ccdg-30day-temp/dking/{uuid}/total-final-result.mt'

    if hl.hadoop_exists(final_mt_path):
        print(' >> skipping final MT creation')
    else:
        print(' >> running final MT creation')

        if prefix_rows is not None:
            partitions = (prefix_rows + 1500) // 1500

        b = hb.Batch(
            default_python_image='hailgenetics/hail:0.2.77',
            backend=hb.ServiceBackend(billing_project='hail'),
            name=f'{uuid}: final MT'
        )
        j = b.new_python_job()
        j.cpu(8)
        j.env("PYSPARK_SUBMIT_ARGS", "--driver-memory 24g --executor-memory 24g pyspark-shell")
        j.call(create_final_mt, combined_tsv, final_mt_path, partitions)

        batch_handle = b.run()
        batch_status = batch_handle.status()

        if batch_status['state'] != 'success':
            print(' >> final MT batch failed, halting.')
            return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rerun", action="store_true")
    parser.add_argument('--prefix-rows', type=int, help='number of intervals to aggregate (FOR TESTING ONLY)')
    parser.add_argument('--group-size', type=int, help='number of intervals to aggregate in one Hail Batch job')

    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(async_main(rerun=args.rerun, prefix_rows=args.prefix_rows, group_size=args.group_size))


main()
