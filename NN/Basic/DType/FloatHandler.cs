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
        public bool AreEqual(float a, float b) => a == b;
        public bool IsGraeter(float a, float b) => a > b;
        public bool IsLess(float a, float b) => a < b;
        #endregion

        #region ternary
        public float Add(float a, float b) => a + b;
        public float Substract(float a, float b) => a - b;
        public float Multiply(float a, float b) => a * b;
        public float Divide(float a, float b) => a / b;
        public float Module(float a, float b) => a % b;
        public float MultiplyAndAdd(float a, float b, float dst) => dst + a * b;
        #endregion

        #region convert
#if NET7_0_OR_GREATER
        public static float Convert(float a) => a;
        public static float Convert(int a) => a;
        public static float Convert(bool a) => a ? .1f : .0f;
#else
        public float Convert(float a) => a;
        public float Convert(int a) => a;
        public float Convert(bool a) => a ? .1f : .0f;
#endif
        #endregion
    }
}
