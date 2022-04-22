using Numnet.Native;
using Numnet.Exceptions;

namespace Numnet.Common{
    internal sealed class TensorTypeInfo
    {
        private static readonly Dictionary<Type, TensorTypeInfo> _typeInfoMap;
        private static readonly Dictionary<DType, Type> _dtypeMap;
        private static readonly Dictionary<DType, int> _sizeMap;
        public readonly DType _dtype;
        public readonly int _size;
        /// <summary>
        /// The priority in type conversion. The larger the value is, the higher priority it has.
        /// </summary>
        public readonly int _priority;
        static TensorTypeInfo(){
            _typeInfoMap = new Dictionary<Type, TensorTypeInfo>()
            {
                { typeof(float), new TensorTypeInfo( DType.Float32, sizeof(float), 4) },
                { typeof(int), new TensorTypeInfo( DType.Int32, sizeof(int), 2) },
                { typeof(long), new TensorTypeInfo( DType.Int64, sizeof(long), 3) },
                { typeof(bool), new TensorTypeInfo( DType.Bool, sizeof(bool), 1) },
                { typeof(double), new TensorTypeInfo( DType.Float64, sizeof(double), 5) }
            };
            _dtypeMap = _typeInfoMap.ToDictionary(k => k.Value._dtype, v => v.Key);
            _sizeMap = _typeInfoMap.ToDictionary(k => k.Value._dtype, v => v.Value._size);
        }
        public TensorTypeInfo(DType dtype, int size, int priority){
            _dtype = dtype;
            _size = size;
            _priority = priority;
        }
        public static TensorTypeInfo GetTypeInfo(Type type){
            TensorTypeInfo res;
            if(!_typeInfoMap.TryGetValue(type, out res)){
                throw new UnsupportedTypeException();
            }
            return res;
        }
        public static Type GetTypeInfo(DType type){
            Type res;
            if(!_dtypeMap.TryGetValue(type, out res)){
                throw new UnsupportedTypeException();
            }
            return res;
        }
        public static int GetTypeSize(DType type){
            int res;
            if(!_sizeMap.TryGetValue(type, out res)){
                throw new UnsupportedTypeException();
            }
            return res;
        }

        public static int GetTypeSize(Type type){
            TensorTypeInfo typeInfo;
            if(!_typeInfoMap.TryGetValue(type, out typeInfo)){
                throw new UnsupportedTypeException();
            }
            return typeInfo._size;
        }
    }
}