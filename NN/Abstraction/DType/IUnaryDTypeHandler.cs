using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Abstraction.DType
{
    public interface IUnaryDTypeHandler<T> where T: unmanaged
    {
        //static abstract void LeftShiftInplace(ref T src, int offset);
        //static abstract void RightShiftAddInplace(ref T src, int offset);
        //static abstract void Increase(ref T src);
        //static abstract void Decrease(ref T src);
        static abstract T Zero { get; }
        static abstract T One { get; }
    }
}
