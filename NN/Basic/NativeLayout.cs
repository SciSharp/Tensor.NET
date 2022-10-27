using NN.Native.Abstraction;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Basic
{
    public struct NativeLayout
    {
        // Modify the enumerator if changing it.
        public readonly static int MAX_NDIM = 4;

        internal int _ndim;
        internal int[] _shape;
        internal int[] _stride;

        #region constructor
        public NativeLayout()
        {
            _ndim = 0;
            _shape = new int[MAX_NDIM];
            _stride = new int[MAX_NDIM];
        }
        public NativeLayout(int[] shape)
        {
            _ndim = shape.Length;
            Debug.Assert(_ndim > 0 && _ndim < MAX_NDIM);
            _shape = new int[MAX_NDIM];
            _stride = new int[MAX_NDIM];
            shape.CopyTo(_shape, 0);
            // init contiguous stride
            int s = 1;
            for(int i = 0; i < _ndim; i++)
            {
                _stride[_ndim - i - 1] = s;
                s *= shape[_ndim - i - 1];
            }
        }
        public NativeLayout(int[] shape, int[] stride)
        {
            _ndim = shape.Length;
            Debug.Assert(_ndim > 0 && _ndim < MAX_NDIM);
            _shape = new int[MAX_NDIM];
            _stride = new int[MAX_NDIM];
            shape.CopyTo(_shape, 0);
            stride.CopyTo(_stride, 0);
        }
        public NativeLayout(ref NativeLayout layout)
        {
            _ndim = layout._ndim;
            Debug.Assert(_ndim > 0 && _ndim < MAX_NDIM);
            _shape = new int[MAX_NDIM];
            _stride = new int[MAX_NDIM];
            layout._shape.CopyTo(_shape, 0);
            layout._stride.CopyTo(_stride, 0);
        }
        #endregion

        #region readonly methods
        public readonly bool IsScalar()
        {
            return _ndim == 1 && _shape[0] == 1;
        }
        public readonly bool IsEmpty()
        {
            if (_ndim == 0) return true;
            for(int i = 0; i < _ndim; i++)
            {
                if (_shape[i] == 0) return true;
            }
            return false;
        }
        public readonly bool IsContiguous()
        {
            int expected = 1;
            for(int i = 0; i < _ndim; i++)
            {
                if (_shape[_ndim - i - 1] != 1 && _stride[_ndim - i - 1] != expected) return false;
                expected *= _shape[_ndim - i - 1];
            }
            // empty layout is supposed not to be contiguous.
            return expected != 0;
        }
        public readonly int TotalElemCount()
        {
            int res = 1;
            for (int i = 0; i < _ndim; i++) res *= _shape[i];
            return res;
        }
        public readonly bool IsSameShape(ref NativeLayout other)
        {
            if (_ndim != other._ndim) return false;
            return _shape.AsSpan().SequenceEqual(other._shape.AsSpan());
        }
        public readonly bool IsSameLayout(ref NativeLayout other)
        {
            return IsSameShape(ref other) && _stride.AsSpan().SequenceEqual(other._stride);
        }
        #endregion

        public void RemoveAxisInplace(int axis)
        {
            Debug.Assert(_ndim >= 2 && axis < _ndim);
            _ndim--;
            for(int i = axis; i < _ndim; i++)
            {
                _shape[i] = _shape[i + 1];
                _stride[i] = _stride[i + 1];
            }
        }

        public void AddAxisInplace(int axis, int shape, int stride)
        {
            Debug.Assert(_ndim + 1 <= MAX_NDIM && axis <= _ndim && shape > 0);
            _ndim++;
            for(int i = _ndim - 1; i > axis; i--)
            {
                _shape[i] = _shape[i - 1];
                _stride[i] = _stride[i - 1];
            }
            _shape[axis] = shape;
            _stride[axis] = stride;
        }

        public void AddAxisInplace(int axis) => AddAxisInplace(axis, 1, _stride[axis] * _shape[axis]);

        public override string ToString()
        {
            string res = "(";
            if (_ndim == 0) res += "Scalar";
            else
            {
                res += "shape = {";
                for (int i = 0; i < _ndim; i++)
                {
                    res += _shape[i].ToString();
                    if (i != _ndim - 1) res += ", ";
                }
                res += "}, stride = {";
                for (int i = 0; i < _ndim; i++)
                {
                    res += _stride[i].ToString();
                    if (i != _ndim - 1) res += ", ";
                }
                res += "}";
            }
            res += ")";
            return res;
        }
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Enumerator GetIndexEnumerator(in NativeLayout layout) => new Enumerator(layout);

        public ref struct Enumerator
        {
            private int _index1, _index2, _index3;
            private int _shape1, _shape2, _shape3;
            private int _stride0, _stride1, _stride2, _stride3;
            private int _product1, _product2, _product3;
            private int _ndim;
            private int _currentIndex;

            [MethodImpl (MethodImplOptions.AggressiveInlining)]
            public Enumerator(in NativeLayout layout)
            {
                _ndim = layout._ndim;
                _currentIndex = 0;
                if(_ndim == 4)
                {
                    _index1 = _index2 = _index3 = 0;
                    _shape1 = layout._shape[1];
                    _shape2 = layout._shape[2];
                    _shape3 = layout._shape[3];
                    _stride0 = layout._stride[0];
                    _stride1 = layout._stride[1];
                    _stride2 = layout._stride[2];
                    _stride3 = layout._stride[3];
                    _product1 = _stride1 * _shape1;
                    _product2 = _stride2 * _shape2;
                    _product3 = _stride3 * _shape3;
                }
                else if(_ndim == 3)
                {
                    _index1 = _index2 = 0;
                    _shape1 = layout._shape[1];
                    _shape2 = layout._shape[2];
                    _stride0 = layout._stride[0];
                    _stride1 = layout._stride[1];
                    _stride2 = layout._stride[2];
                    _product1 = _stride1 * _shape1;
                    _product2 = _stride2 * _shape2;
                }
                else if (_ndim == 2)
                {
                    _index1 = 0;
                    _shape1 = layout._shape[1];
                    _stride0 = layout._stride[0];
                    _stride1 = layout._stride[1];
                    _product1 = _stride1 * _shape1;
                }
                else if (_ndim == 1)
                {
                    _stride0 = layout._stride[0];
                }
            }
            /// <summary>
            /// Move the index to next and return the old index value.
            /// Note that there is no check for overflow. It should be used carefully and never be exposed to users.
            /// </summary>
            /// <returns></returns>
            /// <exception cref="NotImplementedException"></exception>
            public int MoveNext()
            {
                return _ndim switch
                {
                    1 => MoveWithNDim1(),
                    2 => MoveWithNDim2(),
                    3 => MoveWithNDim3(),
                    4 => MoveWithNDim4(),
                    _ => throw new NotImplementedException()
                };
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private int MoveWithNDim1()
            {
                var res = _currentIndex;
                _currentIndex += _stride0;
                return res;
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private int MoveWithNDim2()
            {
                var res = _currentIndex;
                _index1++;
                _currentIndex += _stride1;
                if(_index1 >= _shape1)
                {
                    _currentIndex += _stride0 - _product1;
                }
                return res;
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private int MoveWithNDim3()
            {
                var res = _currentIndex;
                _index2++;
                _currentIndex += _stride2;
                if (_index2 >= _shape2)
                {
                    _index1++;
                    _currentIndex += _stride1 - _product2;
                }
                if (_index1 >= _shape1)
                {
                    _currentIndex += _stride0 - _product1;
                }
                return res;
            }
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private int MoveWithNDim4()
            {
                var res = _currentIndex;
                _index3++;
                _currentIndex += _stride3;
                if (_index3 >= _shape3)
                {
                    _index2++;
                    _currentIndex += _stride2 - _product3;
                }
                if (_index2 >= _shape2)
                {
                    _index1++;
                    _currentIndex += _stride1 - _product2;
                }
                if (_index1 >= _shape1)
                {
                    _currentIndex += _stride0 - _product1;
                }
                return res;
            }
        }
    }
}
