using NN.Native.Abstraction.DType;
using System.Runtime.CompilerServices;

namespace NN.Native.Basic.DType
{
    public class FloatHandler: IUnaryDTypeHandler<float>, IBinaryDTypeHandler<float, float>, ITernaryDTypeHandler<float, float, float>, 
            INativeConvertible<int, float>, INativeConvertible<bool, float>, INativeConvertible<float, float>
    {
        #region unary
        public static float Zero { get => .0f; }
        public static float One { get => 1.0f; }
        #endregion

        #region binary
        public static void AddInplace(ref float src, in float addend) => src += addend;
        public static void SubstractInplace(ref float src, in float subtrahend) => src -= subtrahend;
        public static void MultiplyInplace(ref float src, in float multiplicator) => src *= multiplicator;
        public static void DivideInplace(ref float src, in float divisor) => src /= divisor;
        public static void ModuleInplace(ref float src, in float modulus) => src %= modulus;
        //public static void LeftShift(ref float src, ref float dst, int offset) => throw new InvalidOperationException();
        //public static void RightShift(ref float src, ref float dst, int offset) => throw new InvalidOperationException();
        public static bool AreEqual(in float a, in float b) => a == b;
        public static bool IsGraeter(in float a, in float b) => a > b;
        public static bool IsLess(in float a, in float b) => a < b;
        #endregion

        #region ternary
        public static void Add(in float a, in float b, ref float dst) => dst = a + b;
        public static void Substract(in float a, in float b, ref float dst) => dst = a - b;
        public static void Multiply(in float a, in float b, ref float dst) => dst = a * b;
        public static void Divide(in float a, in float b, ref float dst) => dst = a / b;
        public static void Module(in float a, in float b, ref float dst) => dst = a % b;
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float MultiplyAndAdd(float a, float b, float dst) => dst + a * b;
        #endregion

        #region convert
        public static void Convert(in float a, ref float b) => b = a;
        public static void Convert(in int a, ref float b) => b = a;
        public static void Convert(in bool a, ref float b) => b = a ? .1f: .0f;
        #endregion
    }
}
