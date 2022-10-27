using NN.Native.Basic;

namespace NN.Native.Abstraction.Operators
{
    public interface IMatmulOperator<TA, TB, TC>: ITernaryOperator where TA : unmanaged where TB : unmanaged where TC : unmanaged
    {
        static abstract void Exec(ReadOnlySpan<TA> a, ReadOnlySpan<TB> b, Span<TC> c, in NativeLayout LayoutA, in NativeLayout LayoutB, in NativeLayout LayoutC);
    }
}
