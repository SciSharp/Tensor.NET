using NN.Native.Abstraction.Data;
using NN.Native.Basic;
using NN.Native.Extensions;
using NN.Native.Exceptions;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NN.Native.Data
{
    public struct NativeArray<T> where T: unmanaged
    {
        internal NativeStorage<T> _storage;
        public NativeLayout _layout;
        
        public Span<T> Span { get => _storage._memory.Span; }

        public NativeArray(NativeLayout layout, INativeMemoryManager? memoryManager)
        {
            _layout = layout;
            if(memoryManager is not null)
            {
                _storage = new NativeStorage<T>(memoryManager.AllocateMemory<T>(_layout.TotalElemCount()), 0);
            }
            else
            {
                _storage = new NativeStorage<T>(INativeMemoryManager.Default.AllocateMemory<T>(_layout.TotalElemCount()), 0);
            }
        }

        public NativeArray(NativeLayout layout, NativeStorage<T> storage)
        {
            _layout = layout;
            _storage = storage;
        }

        public NativeArray(NativeLayout layout, Memory<T> memory)
        {
            _layout = layout;
            _storage = new NativeStorage<T>(memory, 0);
        }

        public unsafe MemoryHandle Pin()
        {
            return _storage._memory.Pin();
        }

        public readonly bool IsMutable() => _storage._mutable;
        public readonly bool IsMemoryOwner() => _storage._owned;

        public T this[params int[] indices]
        {
            get
            {
                if(indices.Length != _layout._ndim)
                {
                    throw new InvalidShapeException();
                }
                return Span[_layout.IndicesToOffset(indices)];
            }
            set
            {
                if (indices.Length != _layout._ndim)
                {
                    throw new InvalidShapeException();
                }
                Span[_layout.IndicesToOffset(indices)] = value;
            }
        }

        public override string ToString()
        {
            var layout = _layout;
            var indexEnumerator = NativeLayout.GetIndexEnumerator(layout);

            StringBuilder r = new StringBuilder($"NativeArray {layout}:\n");
            var span = Span;
            for (int i = 0; i < layout.TotalElemCount(); i++)
            {
                int mod = 1;
                for (int j = layout._ndim - 1; j >= 0; j--)
                {
                    mod *= layout._shape[j];
                    if (i % mod == 0)
                    {
                        r.Append("[");
                    }
                    else
                    {
                        break;
                    }
                }
                r.Append(" ").Append(span[indexEnumerator.MoveNext()]);

                if ((i + 1) % layout._shape[layout._ndim - 1] != 0) r.Append(",");

                r.Append(" ");
                mod = 1;
                int hit_times = 0;
                for (int j = layout._ndim - 1; j >= 0; j--)
                {
                    mod *= layout._shape[j];
                    if ((i + 1) % mod == 0)
                    {
                        r.Append("]");
                        hit_times++;
                    }
                    else
                    {
                        break;
                    }
                }
                if (hit_times > 0 && hit_times < layout._ndim)
                {
                    r.Append(",\n");
                    for (int j = 0; j < layout._ndim - hit_times; j++)
                    {
                        r.Append(" ");
                    }
                }
            }
            // r.Append("\n");
            return r.ToString();
        }

        public static explicit operator NativeArray<int>(NativeArray<T> src)
        {
            if(typeof(T) != typeof(int))
            {
                throw new BadTypeException();
            }
            return new NativeArray<int>(new NativeLayout(src._layout), (NativeStorage<int>)(ValueType)src._storage);
        }

        public static explicit operator NativeArray<float>(NativeArray<T> src)
        {
            if (typeof(T) != typeof(float))
            {
                throw new BadTypeException();
            }
            return new NativeArray<float>(new NativeLayout(src._layout), (NativeStorage<float>)(ValueType)src._storage);
        }

        public static explicit operator NativeArray<double>(NativeArray<T> src)
        {
            if (typeof(T) != typeof(double))
            {
                throw new BadTypeException();
            }
            return new NativeArray<double>(new NativeLayout(src._layout), (NativeStorage<double>)(ValueType)src._storage);
        }
    }

    public partial class NativeArray
    {
        [method: MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static NativeArray<T> FromArray<T>(NativeLayout layout, T[] data) where T : unmanaged
        {
            return new NativeArray<T>(layout, new Memory<T>(data));
        }
        public static class Random
        {
            public static NativeArray<T> Normal<T>(NativeLayout layout, T minValue, T maxValue) where T: unmanaged
            {
                NativeArray<T> res = new NativeArray<T>(layout, new DefaultNativeMemoryManager());
                if(typeof(T) == typeof(int))
                {
                    var span = MemoryMarshal.Cast<T, int>(res.Span);
                    var random = new System.Random();
                    for(int i = 0; i < span.Length; i++)
                    {
                        span[i] = random.Next((int)(object)minValue, (int)(object)maxValue);
                    }
                }
                else if (typeof(T) == typeof(float))
                {
                    var span = MemoryMarshal.Cast<T, float>(res.Span);
                    var random = new System.Random();
                    for (int i = 0; i < span.Length; i++)
                    {
                        span[i] = ((float)random.NextDouble()) * ((float)(object)maxValue - (float)(object)minValue) + (float)(object)minValue;
                    }
                }
                else if (typeof(T) == typeof(double))
                {
                    var span = MemoryMarshal.Cast<T, double>(res.Span);
                    var random = new System.Random();
                    for (int i = 0; i < span.Length; i++)
                    {
                        span[i] = ((double)random.NextDouble()) * ((double)(object)maxValue - (double)(object)minValue) + (double)(object)minValue;
                    }
                }
                else
                {
                    throw new NotImplementedException();
                }
                return res;
            }
        }
    }
}
