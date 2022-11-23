using NN.Core.Algebra;
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
        public virtual NDArray<T> Matmul(NDArray<T> other)
        {
#if NET7_0_OR_GREATER
            if (this.Ndim <= 2 && other.Ndim <= 2)
            {
                return Matmul<T>.Exec(this, other);
            }
            else
            {
                throw new NotImplementedException();
            }
#else
            throw new NotImplementedException();
#endif
        }
    }
}
