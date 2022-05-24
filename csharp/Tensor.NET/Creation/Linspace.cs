using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Native.Param;

namespace Tensornet{
    public static partial class Tensor{
        /// <summary>
        /// Returns num evenly spaced samples, calculated over the interval [start, stop]. The endpoint of the interval can optionally be excluded.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.linspace.html?highlight=linspace#numpy.linspace
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="start"> The starting value of the sequence. </param>
        /// <param name="stop"> The end value of the sequence, unless endpoint is set to False. In that case, the sequence consists of all but the last of num + 1 evenly spaced samples, so that stop is excluded. Note that the step size changes when endpoint is False. </param>
        /// <param name="num"> Number of samples to generate, which must be non-negative. </param>
        /// <param name="isEndPoint"> If true, stop is the last sample. Otherwise, it is not included. Default is true. </param>
        /// <returns></returns>
        public static Tensor<T> Linspace<T>(double start, double stop, int num, bool isEndPoint = true) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(new TensorShape(num), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            LinspaceInternal<T>(res, start, stop, num, isEndPoint);
            return res;
        }

        private static unsafe void LinspaceInternal<T>(Tensor<T> t, double start, double stop, int num, bool isEndPoint) where T : struct, IConvertible, IEquatable<T>{
            LinspaceParam param = new LinspaceParam() { start = start, stop = stop, num = num, isEndpoint = isEndPoint };
            IntPtr status = NativeExecutor.Execute(NativeApi.Linspace, t.TMemory, t.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
    }
}