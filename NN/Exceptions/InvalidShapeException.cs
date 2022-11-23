using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Exceptions
{
    public class InvalidShapeException : Exception
    {
        public InvalidShapeException(string? message) : base(message) { }
        public InvalidShapeException() : base() { }
    }
}
