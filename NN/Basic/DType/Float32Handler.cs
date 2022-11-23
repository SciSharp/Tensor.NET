using NN.Native.Abstraction.DType;
using System.Runtime.CompilerServices;

namespace NN.Native.Basic.DType
{
    public class Float32Handler: INativeDTypeHandler<float>,
            INativeConvertible<int, float>, INativeConvertible<bool, float>, INativeConvertible<float, float>
    {
        #region unary
        public float Zero { get => .0f; }
        public float One { get => 1.0f; }
        #endregion

        #region binary
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool AreEqual(float a, float b) => a == b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsGraeter(float a, float b) => a > b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsLess(float a, float b) => a < b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float GetNegative(float x) => -x;
        #endregion

        #region ternary
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Add(float a, float b) => a + b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Subtract(float a, float b) => a - b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Multiply(float a, float b) => a * b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Divide(float a, float b) => a / b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Module(float a, float b) => a % b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float MultiplyAndAdd(float a, float b, float dst) => dst + a * b;
        #endregion

        #region convert
#if NET7_0_OR_GREATER
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Convert(float a) => a;
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Convert(int a) => a;
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Convert(bool a) => a ? .1f : .0f;
#else
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Convert(float a) => a;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Convert(int a) => a;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public float Convert(bool a) => a ? .1f : .0f;
#endif
        #endregion
    }
}
