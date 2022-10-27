using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Data
{
    internal struct NativeStorage<T> where T: unmanaged
    {
        internal Memory<T> _memory;
        internal int _offset;
        internal bool _mutable;
        internal bool _owned;

        public NativeStorage(Memory<T> memory, int offset, bool mutable = true, bool owned = true)
        {
            _memory = memory;
            _offset = offset;
            _mutable = mutable;
            _owned = owned;
        }


        public override bool Equals([NotNullWhen(true)] object? obj)
        {
            if (obj is null or not NativeStorage<T>) return false;
            else return this == (NativeStorage<T>)obj;
        }
        public override int GetHashCode()
        {
            return base.GetHashCode() + _offset;
        }

        public static bool operator ==(NativeStorage<T> a, NativeStorage<T> b)
        {
            return a._memory.Equals(b._memory) && a._offset == b._offset;
        }
        public static bool operator !=(NativeStorage<T> a, NativeStorage<T> b)
        {
            return !(a == b);
        }
    }
}
