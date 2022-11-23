using NN.Core.Arithmetic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Core
{
    public partial class NDArray<T>
    {
        #region Virtual Functions
        public virtual NDArray<T> Add(NDArray<T> other)
        {
#if NET7_0_OR_GREATER
            return Arithmetic.BasicArithmetic<T>.Add(this, other);
#else
            throw new NotImplementedException();
#endif
        }
        public virtual NDArray<T> Add(T other)
        {
#if NET7_0_OR_GREATER
            return Arithmetic.BasicArithmetic<T>.Add(this, other);
#else
            throw new NotImplementedException();
#endif
        }
        public virtual NDArray<T> Subtract(NDArray<T> other)
        {
#if NET7_0_OR_GREATER
            return Arithmetic.BasicArithmetic<T>.Subtract(this, other);
#else
            throw new NotImplementedException();
#endif
        }
        public virtual NDArray<T> Subtract(T other)
        {
#if NET7_0_OR_GREATER
            return Arithmetic.BasicArithmetic<T>.Subtract(this, other);
#else
            throw new NotImplementedException();
#endif
        }
        public virtual NDArray<T> Multiply(NDArray<T> other)
        {
#if NET7_0_OR_GREATER
            return Arithmetic.BasicArithmetic<T>.Multiply(this, other);
#else
            throw new NotImplementedException();
#endif
        }
        public virtual NDArray<T> Multiply(T other)
        {
#if NET7_0_OR_GREATER
            return Arithmetic.BasicArithmetic<T>.Multiply(this, other);
#else
            throw new NotImplementedException();
#endif
        }
        public virtual NDArray<T> Divide(NDArray<T> other)
        {
#if NET7_0_OR_GREATER
            return Arithmetic.BasicArithmetic<T>.Divide(this, other);
#else
            throw new NotImplementedException();
#endif
        }
        public virtual NDArray<T> Divide(T other)
        {
#if NET7_0_OR_GREATER
            return Arithmetic.BasicArithmetic<T>.Divide(this, other);
#else
            throw new NotImplementedException();
#endif
        }
        public virtual NDArray<T> BeDivided(T other)
        {
#if NET7_0_OR_GREATER
            return Arithmetic.BasicArithmetic<T>.Divide(other, this);
#else
            throw new NotImplementedException();
#endif
        }
        protected virtual NDArray<T> GetNegative()
        {
#if NET7_0_OR_GREATER
            return BasicArithmetic<T>.Negative(this);
#else
            throw new NotImplementedException();
#endif
        }
        #endregion

        #region Operators
        public static NDArray<T> operator +(NDArray<T> a, NDArray<T> b)
        {
            return a.Add(b);
        }
        public static NDArray<T> operator +(NDArray<T> a, T b)
        {
            return a.Add(b);
        }
        public static NDArray<T> operator +(T a, NDArray<T> b)
        {
            return b.Add(a);
        }

        public static NDArray<T> operator -(NDArray<T> src)
        {
            return src.GetNegative();
        }

        public static NDArray<T> operator -(NDArray<T> a, NDArray<T> b)
        {
            return a.Subtract(b);
        }
        public static NDArray<T> operator -(NDArray<T> a, T b)
        {
            return a.Subtract(b);
        }
        public static NDArray<T> operator -(T a, NDArray<T> b)
        {
            return -b.Subtract(a);
        }

        public static NDArray<T> operator *(NDArray<T> a, NDArray<T> b)
        {
            return a.Matmul(b);
        }
        public static NDArray<T> operator *(NDArray<T> a, T b)
        {
            return a.Multiply(b);
        }
        public static NDArray<T> operator *(T a, NDArray<T> b)
        {
            return b.Multiply(a);
        }

        public static NDArray<T> operator /(NDArray<T> a, NDArray<T> b)
        {
            return a.Divide(b);
        }
        public static NDArray<T> operator /(NDArray<T> a, T b)
        {
            return a.Divide(b);
        }
        public static NDArray<T> operator /(T a, NDArray<T> b)
        {
            return b.BeDivided(a);
        }
        #endregion
    }
}
