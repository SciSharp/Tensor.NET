#!/usr/bin/env bash
set -e

BUILD_TYPE=Release
NN_WITH_TEST=OFF
MAX_PARALLEL_JOBS=0

cpu_number=`nproc`
package_name=libnumnet
output_dir=./output
build_dir=./build/Linux/$BUILD_TYPE

function config_default_max_jobs() {
    if [[ $OS =~ "NT" ]]; then
        ((MAX_PARALLEL_JOBS = ${cpu_number} - 1))
        if [[ ${MAX_PARALLEL_JOBS} -le 0 ]]; then
            MAX_PARALLEL_JOBS=1
        fi
    else
        ((MAX_PARALLEL_JOBS = ${cpu_number} + 2))
    fi
    echo "config default MAX_PARALLEL_JOBS to ${MAX_PARALLEL_JOBS} [cpu number is:${cpu_number}]"
}

config_default_max_jobs
echo "EXTRA_CMAKE_ARGS: ${EXTRA_CMAKE_ARGS}"

function usage() {
    echo "$0 args1 args2 .."
    echo "available args detail:"
    echo "-d : Build with Debug mode, default Release mode"
    echo "-t : Build with test, default without test"
    echo "-j : run N jobs in parallel for ninja, defaut is cpu_number + 2"
    echo "-h : show usage"
    echo "append other cmake config by config EXTRA_CMAKE_ARGS"
    exit -1
}

while getopts "lnsrhdcmve:j:" arg
do
    case $arg in
        j)
            MAX_PARALLEL_JOBS=$OPTARG
            echo "config MAX_PARALLEL_JOBS to ${MAX_PARALLEL_JOBS}"
            ;;
        d)
            echo "Build with Debug mode"
            BUILD_TYPE=Debug
            ;;
        t)
            echo "Build with test"
            NN_WITH_TEST=ON
            ;;
        h)
            echo "show usage"
            usage
            ;;
        ?)
            echo "unkonw argument"
            usage
            ;;
    esac
done
echo "------------------------------------"
echo "build config summary:"
echo "BUILD_TYPE: $BUILD_TYPE"
echo "NN_WITH_TEST: $NN_WITH_TEST"
echo "MAX_PARALLEL_JOBS: $MAX_PARALLEL_JOBS"
echo "------------------------------------"

function cmake_build() {
    build_dir=./build/Linux/$BUILD_TYPE
    # fork a new bash to handle EXTRA_CMAKE_ARGS env with space
    bash -c "rm -rf ${build_dir}"
    mkdir -p $build_dir
    cd "build"
    bash -c "cmake .. -DNN_WITH_TEST=$NN_WITH_TEST -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DNN_CROSS_COMPILE=OFF ${EXTRA_CMAKE_ARGS}"
    bash -c "make -j${MAX_PARALLEL_JOBS}"
    cd ".."
    bash -c "cp ${build_dir}/lib/${package_name}.so ${output_dir}/${package_name}.so"
    echo "copy the library file to output directory."
}

cmake_build