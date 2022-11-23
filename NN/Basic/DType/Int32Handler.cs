using NN.Native.Abstraction.DType;
using System.Runtime.CompilerServices;

namespace NN.Native.Basic.DType
{
    public class Int32Handler : INativeDTypeHandler<int>,INativeConvertible<int, int>, INativeConvertible<bool, int>, INativeConvertible<float, int>
    {
        #region unary
        public int Zero { get => 0; }
        public int One { get => 1; }
        #endregion

        #region binary
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool AreEqual(int a, int b) => a == b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsGraeter(int a, int b) => a > b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsLess(int a, int b) => a < b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetNegative(int x) => -x;
        #endregion

        #region ternary
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Add(int a, int b) => a + b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Subtract(int a, int b) => a - b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Multiply(int a, int b) => a * b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Divide(int a, int b) => a / b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Module(int a, int b) => a % b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int MultiplyAndAdd(int a, int b, int dst) => dst + a * b;
        #endregion

        #region convert
#if NET7_0_OR_GREATER
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Convert(int a) => a;
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Convert(float a) => (int)a;
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Convert(bool a) => a ? 1 : 0;
#else
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Convert(int a) => a;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Convert(float a) => (int)a;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Convert(bool a) => a ? 1 : 0;
#endif
        #endregion
    }
}
