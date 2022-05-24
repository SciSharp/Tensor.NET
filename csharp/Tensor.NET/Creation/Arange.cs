using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;
using Tensornet.Native.Param;

namespace Tensornet{
    public static partial class Tensor{
        /// <summary>
        /// Return evenly spaced values within a given interval. Values are generated within the half-open interval [0, stop) with step 1.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.arange.html?highlight=arange#numpy.arange
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="stop"> The upper bound of the range of generation. </param>
        /// <returns></returns>
        public static Tensor<T> Arange<T>(double stop) where T : struct, IConvertible, IEquatable<T>{
            int num = (int)System.Math.Ceiling(stop);
            Tensor<T> res = new Tensor<T>(new TensorShape(num), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            ArangeInternal<T>(res, 0, stop, 1);
            return res;
        }
        /// <summary>
        /// Return evenly spaced values within a given interval. Values are generated within the half-open interval [start, stop) with step 1.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.arange.html?highlight=arange#numpy.arange
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="start"> The lower bound of the range of generation. </param>
        /// <param name="stop"> The upper bound of the range of generation. </param>
        /// <returns></returns>
        public static Tensor<T> Arange<T>(double start, double stop) where T : struct, IConvertible, IEquatable<T>{
            int num = (int)System.Math.Ceiling(stop - start);
            Tensor<T> res = new Tensor<T>(new TensorShape(num), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            ArangeInternal<T>(res, start, stop, 1);
            return res;
        }
        /// <summary>
        /// Return evenly spaced values within a given interval. Values are generated within the half-open interval [start, stop) with the given step.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.arange.html?highlight=arange#numpy.arange
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="start"> The lower bound of the range of generation. </param>
        /// <param name="stop"> The upper bound of the range of generation. </param>
        /// <param name="step"> The step of the generation. </param>
        /// <returns></returns>
        public static Tensor<T> Arange<T>(double start, double stop, double step) where T : struct, IConvertible, IEquatable<T>{
            if(step <= 0){
                throw new InvalidParamException("The step of arrange should be a positive number.");
            }
            int num = (int)System.Math.Ceiling((stop - start) / step);
            Tensor<T> res = new Tensor<T>(new TensorShape(num), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            ArangeInternal<T>(res, start, stop, step);
            return res;
        }

        private static unsafe void ArangeInternal<T>(Tensor<T> t, double start, double stop, double step) where T : struct, IConvertible, IEquatable<T>{
            ArangeParam param = new ArangeParam() { start = start, stop = stop, step = step};
            IntPtr status = NativeExecutor.Execute(NativeApi.Arange, t.TMemory, t.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
    }
}