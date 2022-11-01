using NN.Native.Abstraction.Data;
using NN.Native.Basic;
using System;
using System.Collections.Generic;
using System.Linq;
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

        public NativeArray(NativeLayout layout, INativeMemoryManager memoryManager)
        {
            _layout = layout;
            _storage = new NativeStorage<T>(memoryManager.AllocateMemory<T>(_layout.TotalElemCount()), 0);
        }

        public NativeArray(NativeLayout layout, Memory<T> memory)
        {
            _layout = layout;
            _storage = new NativeStorage<T>(memory, 0);
        }

        public readonly bool IsMutable() => _storage._mutable;
        public readonly bool IsMemoryOwner() => _storage._owned;
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
                else
                {
                    throw new NotImplementedException();
                }
                return res;
            }
        }
    }
}
