using Numnet.Native;

namespace Numnet.Base{
    internal sealed class TensorTypeInfo
    {
        private static readonly Dictionary<Type, TensorTypeInfo> _typeInfoMap;
        private static readonly Dictionary<DType, Type> _dtypeMap;
        public readonly DType _dtype;
        public readonly int _size;
        static TensorTypeInfo(){
            _typeInfoMap = new Dictionary<Type, TensorTypeInfo>()
            {
                { typeof(float), new TensorTypeInfo( DType.Float32, sizeof(float)) },
                { typeof(int), new TensorTypeInfo( DType.Int32, sizeof(int)) },
                { typeof(long), new TensorTypeInfo( DType.Int64, sizeof(long)) },
                { typeof(bool), new TensorTypeInfo( DType.Bool, sizeof(bool)) },
                { typeof(double), new TensorTypeInfo( DType.Float64, sizeof(double)) }
            };
            _dtypeMap = _typeInfoMap.ToDictionary(k => k.Value._dtype, v => v.Key);
        }
        public TensorTypeInfo(DType dtype, int size){
            _dtype = dtype;
            _size = size;
        }
        public static TensorTypeInfo GetTypeInfo(Type type){
            TensorTypeInfo res;
            if(!_typeInfoMap.TryGetValue(type, out res)){
                throw new UnsopportedTypeException();
            }
            return res;
        }
        public static Type GetTypeInfo(DType type){
            Type res;
            if(!_dtypeMap.TryGetValue(type, out res)){
                throw new UnsopportedTypeException();
            }
            return res;
        }
    }
}