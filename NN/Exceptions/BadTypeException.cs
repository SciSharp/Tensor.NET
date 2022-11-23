using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Exceptions
{
    public class BadTypeException : Exception
    {
        public BadTypeException(string? message) : base(message) { }
        public BadTypeException() : base() { }
    }
}
