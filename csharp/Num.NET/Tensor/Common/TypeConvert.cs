using Numnet.Native;
using Numnet.Common;
using Numnet.Native.Param;

namespace Numnet{
    public partial class Tensor<T> where T : struct, IEquatable<T>, IConvertible{
        public Tensor<TD> ToTensor<TD>() where TD : struct, IEquatable<TD>, IConvertible{
            Tensor<TD> res = new Tensor<TD>(new TensorLayout(TLayout, TensorTypeInfo.GetTypeInfo(typeof(TD))._dtype));
            Console.WriteLine($"res dim : {res.Dim}");
            TypeConvertInternal(TMemory, res.TMemory, TLayout, res.TLayout);
            return res;
        }
        private static unsafe void TypeConvertInternal(ITensorMemory mA, ITensorMemory mB, TensorLayout lA, TensorLayout lB){
            TypeConvertParam param = new TypeConvertParam() { targetType = lB.DType };
            IntPtr status = NativeExecutor.Execute(NativeApi.TypeConvert, mA, mB, lA, lB, new IntPtr(&param), Tensor<T>.Provider);
            NativeStatus.AssertOK(status);
        }
    }
}