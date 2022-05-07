using Numnet.Common;
using Numnet.Native;
using Numnet.Native.Param;

namespace Numnet{
    public static partial class Tensor{
        public static partial class Random{
            /// <summary>
            /// Genetate a randome Tensor with value in range [minValue, maxValue)
            /// </summary>
            /// <typeparam name="T"></typeparam>
            /// <param name="shape"></param>
            /// <param name="minValue"></param>
            /// <param name="maxValue"></param>
            /// <returns></returns>
            public static Tensor<T> Uniform<T>(TensorShape shape, double minValue = .0, double maxValue = 1.0) where T : struct, IConvertible, IEquatable<T>{
                Tensor<T> res = new Tensor<T>(shape, TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
                UniformInternal<T>(res, minValue, maxValue);
                return res;
            }
            private unsafe static void UniformInternal<T>(Tensor<T> t, double minValue, double maxValue) where T : struct, IConvertible, IEquatable<T>{
                UniformParam param = new UniformParam() { minValue = minValue, maxValue = maxValue };
                IntPtr status = NativeExecutor.Execute(NativeApi.Uniform, t.TMemory, t.TLayout, new IntPtr(&param), Tensor<T>.Provider);
                NativeStatus.AssertOK(status);
            }
        }
    }
}