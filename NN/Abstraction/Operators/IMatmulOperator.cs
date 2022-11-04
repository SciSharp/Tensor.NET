using NN.Native.Basic;

namespace NN.Native.Abstraction.Operators
{
    public unsafe interface IMatmulOperator<T>: ITernaryOperator where T : unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract void Exec(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, in NativeLayout LayoutA, in NativeLayout LayoutB, in NativeLayout LayoutC);
#else
        void Exec(ReadOnlySpan<T> a, ReadOnlySpan<T> b, Span<T> c, in NativeLayout LayoutA, in NativeLayout LayoutB, in NativeLayout LayoutC);
#endif
    }
}
