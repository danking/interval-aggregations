from typing import List, Tuple, Optional, Callable
import hail as hl
import hailtop.batch as hb
import functools
import asyncio
import json
import argparse
import math

from hailtop.utils import grouped, secret_alnum_string, bounded_gather
from hailtop.aiocloud.aiogoogle import GoogleStorageAsyncFS
from hailtop.aiotools.fs import AsyncFS
from hailtop.utils import tqdm
from hailtop.utils.utils import digits_needed


def mb_per_cpu_per_memory(memory: str) -> float:
    """How much memory, in MB, should Hail use for each type of machine.

    These numbers are slightly less than the full memory available to a 1 core job on the given
    machine type.
    """
    if memory == 'highmem':
        return 6000
    if memory == 'standard':
        return 3250
    return 700


def aggregate_by_intervals(index: int,
                           cpus: int,
                           intervals: List[hl.Interval],
                           uuid: str) -> List[str]:
    """Compute the interval statistics for a group of intervals.

    We group many intervals into one Job in order to benefit from JVM JIT compilation as well as to
    decrease the overhead of reading metadata from the VDS file.

    We return a list of the successfully written paths. This is necessary because a Batch Job might
    get preempted by Google. When the job is restarted, we regenerate resuts for every interval in
    the group using new filenames just in case one of the old files was corrupted by the preemption.

    """
    import hail as hl

    print(f'hl.init(master="local[{cpus}]")')
    hl.init(master=f'local[{cpus}]')

    exomes_vds = hl.vds.read_vds('gs://ccdg/vds/split_200k_ccdg_exome.vds')
    exomes_ref_mt = exomes_vds.reference_data
    exomes_ref_mt = exomes_ref_mt.annotate_entries(base_cnt = exomes_ref_mt.END - exomes_ref_mt.locus.position)

    paths = [
        aggregate_interval_to_file(exomes_ref_mt, interval, index, uuid)
        for interval in intervals]
    return paths


def aggregate_interval_to_file(exomes_ref_mt: hl.MatrixTable,
                               the_interval: hl.Interval,
                               index: int,
                               uuid: str) -> str:
    """Compute the interval statistics for `the_interval`.

    Generate a unique filename and compute the statistics for `the_interval`. We play a silly game
    with `annotate_cols`, `head`, and `select_entries` to avoid using `group_rows_by`. Aggregating
    to column fields over all rows in the matrix table is correct because we filter the matrix table
    to just those rows in the interval.

    """
    import hail as hl
    import sys
    from hailtop.utils import secret_alnum_string

    attempt_id = secret_alnum_string(5)
    path = f'gs://ccdg-30day-temp/dking/{uuid}/output_{index}_{attempt_id}.tsv'
    print(f'writing {the_interval} to {path}', flush=True)
    the_interval = hl.literal(the_interval)
    exomes_ref_mt = exomes_ref_mt.filter_rows(the_interval.contains(exomes_ref_mt.locus))
    ## We must avoid a shuffle which would consume unnecessary amounts of memory so we rewrite the
    ## following to use annotate_cols, knowing that we have already filtered to the one interval of
    ## interest.
    #
    # exomes_ref_mt.group_rows_by(
    #     interval=the_interval
    # ).aggregate(
    #     n_bases = hl.agg.sum(exomes_ref_mt.base_cnt)
    # ).key_rows_by(
    #     interval=the_interval
    # ).n_bases.export(
    #     path
    # )
    exomes_ref_mt = exomes_ref_mt.annotate_cols(
        n_bases_col = hl.agg.sum(exomes_ref_mt.base_cnt)
    )
    exomes_ref_mt = exomes_ref_mt.head(1)
    exomes_ref_mt = exomes_ref_mt.select_entries(
        n_bases = exomes_ref_mt.n_bases_col
    )
    exomes_ref_mt = exomes_ref_mt.key_rows_by(
        interval=the_interval
    )
    exomes_ref_mt.n_bases.export(path)
    return path


def create_final_mt(combined_tsv: str, final_mt_path: str, cpus: int):
    """Convert a TSV of results into a MatrixTable.
    """
    import hail as hl
    print(f'hl.init(master="local[{cpus}]")')
    hl.init(master=f'local[{cpus}]')
    mt = hl.import_matrix_table(combined_tsv, row_fields={'interval': hl.tstr})
    mt = mt.annotate_rows(parsed_interval=hl.parse_locus_interval(mt.interval, reference_genome='GRCh38'))
    mt = mt.key_rows_by('parsed_interval')
    mt = mt.drop('interval')
    mt = mt.rename({'parsed_interval': 'interval', 'x': 'n_bases'})
    mt.write(final_mt_path)


async def construct_aggregation_batch(service_backend: hb.ServiceBackend,
                                      grouped_intervals: List,
                                      cpus: int,
                                      memory: str,
                                      uuid: str,
                                      file_pbar) -> Tuple[hb.Batch, List[str]]:
    """Construct a batch of calls to `aggregate_by_intervals`, one for each group in `grouped_intervals`."""
    b = hb.Batch(
        default_python_image='hailgenetics/hail:0.2.77',
        backend=service_backend,
        name=f'{uuid}: interval statistics'
    )

    async with GoogleStorageAsyncFS() as fs:
        result_paths = await bounded_gather(
            *[functools.partial(construct_aggregation_job, group, cpus, memory, uuid, b, fs, file_pbar)
              for group in grouped_intervals],
            parallelism=150)

    return b, result_paths


async def construct_aggregation_job(group,
                                    cpus: int,
                                    memory: str,
                                    uuid: str,
                                    b: hb.Batch,
                                    fs: AsyncFS,
                                    file_pbar) -> str:
    """Create a job to run `aggregate_by_intervals` if its output does not already exist.

    Each call to `aggregate_by_intervals` returns a list of the successful TSVs. We convert this
    list to JSON and store it in a file in GCS. If this file already exists, then we know this job
    has already succeeded and we can skip it.

    We use `file_pbar` to indicate how many output files we've already checked for existence.

    """
    index = group.group_index
    intervals = group.intervals
    result_path = f'gs://ccdg-30day-temp/dking/{uuid}/successful_paths_{index}.json'

    if not await fs.exists(result_path):
        j = b.new_python_job(attributes={
            'intervals': json.dumps([str(i) for i in intervals]),
            'group_index': str(index)
        })
        j.cpu(cpus)
        j.memory(memory)
        memory_in_mb = cpus * mb_per_cpu_per_memory(memory)
        j.env("PYSPARK_SUBMIT_ARGS", f'--driver-memory {memory_in_mb}m --executor-memory {memory_in_mb}m pyspark-shell')
        result = j.call(aggregate_by_intervals, index, cpus, intervals, uuid)
        b.write_output(result.as_json(), result_path)

    file_pbar.update(1)

    return result_path


def batch_combine2(base_combop: Callable[[hb.job.BashJob, List[str], str], None],
                   combop: Callable[[hb.job.BashJob, List[str], str], None],
                   b: hb.Batch,
                   name: str,
                   paths: List[str],
                   final_location: str,
                   branching_factor: int = 100,
                   suffix: Optional[str] = None):
    """A hierarchical merge using Batch jobs.

    We combine at most `branching_factor` paths at a time. The first layer is given by
    `paths`. Layer n combines the files produced by layer n-1.

    For the first layer, we use `base_combop` to construct a job to combine a given group. For
    subsequent layers, we use `combop`. This permits us, for example, to start with uncompressed
    files, but use compressed files for the intermediate steps and the final step.

    The fully combined (single) file is written to `final_location`.
    """
    assert isinstance(branching_factor, int) and branching_factor >= 1
    n_levels = math.ceil(math.log(len(paths), branching_factor))
    level_digits = digits_needed(n_levels)
    branch_digits = digits_needed((len(paths) + branching_factor) // branching_factor)
    assert isinstance(b._backend, hb.ServiceBackend)
    tmpdir = b._backend.remote_tmpdir.rstrip('/')

    def make_job_and_path(level: int,
                          i: int,
                          make_commands: Callable[[hb.job.BashJob, List[str], str], None],
                          paths: List[str],
                          dependencies: List[hb.job.BashJob],
                          ofile: Optional[str] = None):
        if ofile is None:
            ofile = f'{tmpdir}/{level:0{level_digits}}/{i:0{branch_digits}}'
            if suffix:
                ofile += suffix
        j = b.new_job(name=f'{name}-{level:0{level_digits}}-{i:0{branch_digits}}')
        for d in dependencies:
            j.depends_on(d)
        make_commands(j, paths, ofile)
        return (j, ofile)

    assert n_levels > 0

    if n_levels == 1:
        return make_job_and_path(0, 0, base_combop, paths, [], final_location)

    jobs_and_paths = [
        make_job_and_path(0, i, base_combop, paths, [])
        for i, paths in enumerate(grouped(branching_factor, paths))]

    for level in range(1, n_levels - 1):
        jobs = [x[0] for x in jobs_and_paths]
        paths = [x[1] for x in jobs_and_paths]
        jobs_and_paths = [
            make_job_and_path(level, i, combop, paths, jobs)
            for i, jobs_and_paths in enumerate(grouped(branching_factor, jobs_and_paths))]

    jobs = [x[0] for x in jobs_and_paths]
    paths = [x[1] for x in jobs_and_paths]
    make_job_and_path(n_levels - 1, 0, combop, paths, jobs, final_location)


async def async_main(billing_project: str,
                     remote_tmpdir: str,
                     rerun: bool = False,
                     prefix_rows: Optional[int] = None,
                     group_size: Optional[int] = None,
                     cpus: Optional[int] = None,
                     memory: Optional[str] = None,
                     branching_factor: Optional[int] = None):
    """Run an interval analysis in three steps: hail-query-per-interval, combine-tsvs, convert-to-mt

    This code can restart after most partial batch failures. It tracks metadata about the process in
    a Hail Table stored at `./grouped-intervals.t`. If you want to start fresh, I recommend creating
    a new directory and executing this file from within that directory.

    """
    service_backend = hb.ServiceBackend(billing_project=billing_project, remote_tmpdir=remote_tmpdir)

    if cpus is None:
        cpus = 2
    assert cpus >= 2  # must be at least 2, for some large intervals 4, 8, or 16 may be necessary

    if memory is None:
        memory = 'standard'

    ###########################################################################

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
            prefix_rows=prefix_rows if prefix_rows is not None else hl.missing(hl.tint32),
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

    assert prefix_rows is None or prefix_rows == grouped_intervals_table.prefix_rows.collect()[0]
    assert group_size is None or group_size == grouped_intervals_table.group_size.collect()[0]

    ###########################################################################

    print(f'uuid: {uuid}; cpus: {cpus}; memory: {memory}; location: gs://ccdg-30day-temp/dking/{uuid}')

    ###########################################################################

    with tqdm(desc='checking for extant aggregation files', leave=False, unit='file', total=len(grouped_intervals)) as file_pbar:
         b, result_paths = await construct_aggregation_batch(
             service_backend, grouped_intervals, cpus, memory, uuid, file_pbar)
    if len(b._jobs) == 0:
        print(' >> skipping aggregations')
    else:
        print(' >> running aggregations')
        batch_handle = b.run(wait=False)
        try:
            batch_handle.wait()
        except KeyboardInterrupt:
            batch_handle.cancel()
        batch_status = batch_handle.status()

        if batch_status['state'] != 'success':
            print(' >> first batch failed, halting.')
            return

    ###########################################################################

    combined_tsv = f'gs://ccdg-30day-temp/dking/{uuid}/total-final-result.tsv.bgz'

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
            default_image='gcr.io/hail-vdc/dk/hail:0.2.77',
            backend=service_backend,
            name=f'{uuid}: combine tsvs'
        )

        def combine_tsvs_with_headers(j: hb.job.BashJob, xs: List[str], ofile: str):
            j.command('set -ex -o pipefail')
            j.command('''
function retry() {
    "$@" || { sleep 2 && "$@" ; } || { sleep 5 && "$@" ; }
}''')
            j.command('retry gcloud auth activate-service-account --key-file=/gsa-key/key.json')
            serially_read_tail_of_files_to_stdout = " && ".join([
                f'gsutil -m cat {x} | tail -n +2 -q' for x in xs[1:]])
            j.command(f'''
join-files() {{
    rm -f sink
    mkfifo sink
    ( {{ gsutil -m cat {xs[0]} &&
         {serially_read_tail_of_files_to_stdout}
      }} | bgzip > sink
    ) & pid=$!
    gsutil -m cp sink {ofile}
    wait $pid
}}

retry join-files
''')

        def combine_compressed_tsvs_with_headers(j: hb.job.BashJob, xs: List[str], ofile: str):
            """Combine many compressed TSVs into one compressed TSV.

            We use a named sink to link the subsequent reads of each file with the single output
            file. The first file is read in its entireity. The subsequent files have their header
            removed.

            """
            j.command('set -ex -o pipefail')
            j.command('''
function retry() {
    "$@" || { sleep 2 && "$@" ; } || { sleep 5 && "$@" ; }
}''')
            j.command('retry gcloud auth activate-service-account --key-file=/gsa-key/key.json')
            serially_read_tail_of_files_to_stdout = " && ".join([
                f'gsutil -m cat {x} | bgzip -d | tail -n +2 -q' for x in xs[1:]])
            j.command(f'''
join-files() {{
    rm -f sink
    mkfifo sink
    ( {{ gsutil -m cat {xs[0]} | bgzip -d &&
         {serially_read_tail_of_files_to_stdout}
      }} | bgzip > sink
    ) & pid=$!
    gsutil -m cp sink {ofile}
    wait $pid
}}

retry join-files
''')

        if branching_factor is None:
            branching_factor = 25

        combined = batch_combine2(
            combine_tsvs_with_headers,
            combine_compressed_tsvs_with_headers,
            b,
            'combine-tsvs-with-headers',
            mt_paths,
            final_location=combined_tsv,
            branching_factor=branching_factor,
            suffix='.tsv.bgz')

        batch_handle = b.run(wait=False)
        try:
            batch_handle.wait()
        except KeyboardInterrupt:
            batch_handle.cancel()
        batch_status = batch_handle.status()

        if batch_status['state'] != 'success':
            print(' >> combine batch failed, halting.')
            return

    ###########################################################################

    final_mt_path = f'gs://ccdg-30day-temp/dking/{uuid}/total-final-result.mt'

    if hl.hadoop_exists(final_mt_path):
        print(' >> skipping final MT creation')
    else:
        print(' >> running final MT creation')

        b = hb.Batch(
            default_python_image='hailgenetics/hail:0.2.77',
            backend=service_backend,
            name=f'{uuid}: final MT'
        )
        j = b.new_python_job()
        cpus = 8
        j.cpu(cpus)
        memory_in_mb = cpus * mb_per_cpu_per_memory(memory)
        j.env("PYSPARK_SUBMIT_ARGS", f'--driver-memory {memory_in_mb}m --executor-memory {memory_in_mb}m pyspark-shell')
        j.call(create_final_mt, combined_tsv, final_mt_path, cpus)

        batch_handle = b.run(wait=False)
        try:
            batch_handle.wait()
        except KeyboardInterrupt:
            batch_handle.cancel()
        batch_status = batch_handle.status()

        if batch_status['state'] != 'success':
            print(' >> final MT batch failed, halting.')
            return


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rerun", action="store_true")
    parser.add_argument('--billing-project', type=str, required=True, help='Hail Batch billing project')
    parser.add_argument('--remote-tmpdir', type=str, required=True, help='Hail Batch remote temporary directory')
    parser.add_argument('--prefix-rows', type=int, help='number of intervals to aggregate (FOR TESTING ONLY)')
    parser.add_argument('--group-size', type=int, help='number of intervals to aggregate in one Hail Batch job')
    parser.add_argument('--cpus', type=int, help='number of CPUs to use in aggregate jobs')
    parser.add_argument('--memory', type=str, help='memory to use in aggregate jobs')
    parser.add_argument('--branching-factor', type=int, help='memory to use in aggregate jobs')

    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(
        async_main(billing_project=args.billing_project,
                   remote_tmpdir=args.remote_tmpdir,
                   rerun=args.rerun,
                   prefix_rows=args.prefix_rows,
                   group_size=args.group_size,
                   cpus=args.cpus,
                   memory=args.memory,
                   branching_factor=args.branching_factor))


main()
