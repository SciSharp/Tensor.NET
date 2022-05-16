using Tensornet.Native;
using Tensornet.Common;
using Tensornet.Native.Param;

namespace Tensornet{
    public partial class Tensor<T>{
        public Tensor<TD> ToTensor<TD>() where TD : struct, IEquatable<TD>, IConvertible{
            Tensor<TD> res = new Tensor<TD>(new TensorLayout(TLayout, TensorTypeInfo.GetTypeInfo(typeof(TD))._dtype));
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