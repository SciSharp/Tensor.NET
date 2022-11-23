using NN.Core.Statistics;
using NN.Native.Abstraction.Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Core
{
    public partial class NDArray<T>
    {
        public virtual NDArray<T> Sum(params int[] axis)
        {
            ReduceParam param;
            if(axis.Length == 0)
            {
                int[] realAxis = new int[this.Ndim];
                for (int i = 0; i < this.Ndim; i++) realAxis[i] = i;
                param = new ReduceParam(realAxis);
            }
            else
            {
                param = new ReduceParam(axis);
            }
#if NET7_0_OR_GREATER
            return Reduce<T>.Sum(this, param);
#else
            throw new NotImplementedException();
#endif
        }

        public virtual NDArray<T> Mean(params int[] axis)
        {
            ReduceParam param;
            if (axis.Length == 0)
            {
                int[] realAxis = new int[this.Ndim];
                for (int i = 0; i < this.Ndim; i++) realAxis[i] = i;
                param = new ReduceParam(realAxis);
            }
            else
            {
                param = new ReduceParam(axis);
            }
#if NET7_0_OR_GREATER
            return Reduce<T>.Mean(this, param);
#else
            throw new NotImplementedException();
#endif
        }

        public virtual NDArray<T> Max(params int[] axis)
        {
            ReduceParam param;
            if (axis.Length == 0)
            {
                int[] realAxis = new int[this.Ndim];
                for (int i = 0; i < this.Ndim; i++) realAxis[i] = i;
                param = new ReduceParam(realAxis);
            }
            else
            {
                param = new ReduceParam(axis);
            }
#if NET7_0_OR_GREATER
            return Reduce<T>.Max(this, param);
#else
            throw new NotImplementedException();
#endif
        }

        public virtual NDArray<T> Min(params int[] axis)
        {
            ReduceParam param;
            if (axis.Length == 0)
            {
                int[] realAxis = new int[this.Ndim];
                for (int i = 0; i < this.Ndim; i++) realAxis[i] = i;
                param = new ReduceParam(realAxis);
            }
            else
            {
                param = new ReduceParam(axis);
            }
#if NET7_0_OR_GREATER
            return Reduce<T>.Min(this, param);
#else
            throw new NotImplementedException();
#endif
        }
    }
}
