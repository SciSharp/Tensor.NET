using NN.Native.Basic;

namespace NN.Native.Abstraction.Operators
{
    public unsafe interface IMatmulOperator<T>: ITernaryOperator where T : unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract void Exec(T* a, T* b, T* c, in NativeLayout LayoutA, in NativeLayout LayoutB, in NativeLayout LayoutC);
#else
        void Exec(T* a, T* b, T* c, in NativeLayout LayoutA, in NativeLayout LayoutB, in NativeLayout LayoutC);
#endif
    }
}
