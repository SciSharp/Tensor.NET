
using NN.Native.Basic;
using NN.Native.Operators.Common.Params;

namespace NN.Native.Abstraction.Operators
{
    public interface ITypeConvertOperator<TA, TB>: IBinaryOperator where TA: unmanaged where TB : unmanaged
    {
        static abstract void Exec(ReadOnlySpan<TA> a, Span<TB> b, in NativeLayout layoutA, in NativeLayout layoutB, in TypeConvertParam param);
    }
}
