using NN.Native.Abstraction.DType;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Basic.DType
{
    public class Float64Handler : INativeDTypeHandler<double>,
            INativeConvertible<int, double>, INativeConvertible<bool, double>, INativeConvertible<float, double>, INativeConvertible<double, double>
    {
        #region unary
        public double Zero { get => .0; }
        public double One { get => 1.0; }
        #endregion

        #region binary
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool AreEqual(double a, double b) => a == b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsGraeter(double a, double b) => a > b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsLess(double a, double b) => a < b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double GetNegative(double x) => -x;
        #endregion

        #region ternary
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Add(double a, double b) => a + b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Subtract(double a, double b) => a - b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Multiply(double a, double b) => a * b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Divide(double a, double b) => a / b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Module(double a, double b) => a % b;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double MultiplyAndAdd(double a, double b, double dst) => dst + a * b;
        #endregion

        #region convert
#if NET7_0_OR_GREATER
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Convert(double a) => a;
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Convert(int a) => a;
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Convert(float a) => a;
        [method:MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Convert(bool a) => a ? .1f : .0f;
#else
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Convert(double a) => a;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Convert(int a) => a;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Convert(float a) => a;
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public double Convert(bool a) => a ? .1f : .0f;
#endif
        #endregion
    }
}
