using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Native.Param;

namespace Tensornet{
    public static partial class Tensor{
        /// <summary>
        /// Return a 2-D tensor with ones on the diagonal and zeros elsewhere.
        /// For details, please refer to https://numpy.org/doc/stable/reference/generated/numpy.eye.html?highlight=eye#numpy.eye
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="rows"> Number of rows in the output. </param>
        /// <param name="cols"> Number of columns in the output. </param>
        /// <param name="k"> Index of the diagonal: 0 (the default) refers to the main diagonal, a positive value refers to an upper diagonal, and a negative value to a lower diagonal. </param>
        /// <returns></returns>
        public static Tensor<T> Eye<T>(int rows, int cols, int k) where T : struct, IConvertible, IEquatable<T>{
            Tensor<T> res = new Tensor<T>(new TensorShape(rows, cols), TensorTypeInfo.GetTypeInfo(typeof(T))._dtype);
            EyeInternal<T>(res, k);
            return res;
        }

        private static unsafe void EyeInternal<T>(Tensor<T> t, int k) where T : struct, IConvertible, IEquatable<T>{
            EyeParam param = new EyeParam() { k = k };
            IntPtr status = NativeExecutor.Execute(NativeApi.Eye, t.TMemory, t.TLayout, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
    }
}