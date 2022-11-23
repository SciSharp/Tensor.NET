using NN.Native.Abstraction.Data;
using NN.Native.Basic;
using NN.Native.Data;
using NN.Native.Exceptions;

namespace NN.Native.Abstraction.Operators.Arithmetic
{
    public interface ISubtractOperator<T> : ITernaryOperator where T : unmanaged
    {
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<T> Exec(in NativeArray<T> a, in NativeArray<T> b, INativeMemoryManager? memoryManager = null);
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<T> Exec(in NativeArray<T> a, T b, INativeMemoryManager? memoryManager = null);
#if NET7_0_OR_GREATER
        static abstract
#endif
        NativeArray<T> Exec(T a, in NativeArray<T> b, INativeMemoryManager? memoryManager = null);

        public static NativeLayout DeduceLayout(in NativeLayout a, in NativeLayout b)
        {
            if (!a.IsSameShape(b))
            {
                throw new InvalidShapeException();
            }
            return NativeLayout.ShapeLike(a);
        }
        public static NativeLayout DeduceLayout(in NativeLayout a)
        {
            return NativeLayout.ShapeLike(a);
        }
    }
}
