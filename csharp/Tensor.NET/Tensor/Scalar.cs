using Tensornet.Native;
using Tensornet.Common;
using Tensornet.Exceptions;

namespace Tensornet{
    public class Scalar : IEquatable<Scalar>{
        public DType DataType{ get; internal set; }
        public Scalar<T> AsScalar<T>() where T : struct{
            return new Scalar<T>(GetValue<T>());
        }
        public virtual T GetValue<T>() where T : struct{
            throw new NotImplementedException();
        }
        public virtual void SetValue<T>(T value){
            throw new NotImplementedException();
        }
        public static implicit operator Scalar(int value){
            return new Scalar<int>(value);
        }
        public static implicit operator Scalar(long value){
            return new Scalar<long>(value);
        }
        public static implicit operator Scalar(float value){
            return new Scalar<float>(value);
        }
        public static implicit operator Scalar(double value){
            return new Scalar<double>(value);
        }
        public static implicit operator Scalar(bool value){
            return new Scalar<bool>(value);
        }
        public override bool Equals(object? obj)
        {
            if (obj is null || GetType() != obj.GetType())
            {
                return false;
            }
            if (ReferenceEquals(this, obj)) return true;
            return Equals((Scalar) obj);
        }
        public bool Equals(Scalar? other)
        {
            if(other is null || DataType != other.DataType){
                return false;
            }
            return this == other;
        }
        
        // override object.GetHashCode
        public override int GetHashCode()
        {
            int valueHash = DataType switch
            {
                DType.Bool => GetValue<bool>().GetHashCode() % 0x3f3f3f3f,
                DType.Int32 => GetValue<int>().GetHashCode() % 0x3f3f3f3f,
                DType.Int64 => GetValue<long>().GetHashCode() % 0x3f3f3f3f,
                DType.Float32 => GetValue<float>().GetHashCode() % 0x3f3f3f3f,
                DType.Float64 => GetValue<double>().GetHashCode() % 0x3f3f3f3f,
                _ => throw new NotImplementedException()
            };
            return (valueHash + (int)DataType * 0x0cfcfcfc) % 0x3f3f3f3f;
        }
        public static Scalar operator +(Scalar lhs, Scalar rhs){
            if(lhs.DataType == DType.Float64 || rhs.DataType == DType.Float64){
                return new Scalar<double>(lhs.GetValue<double>() + rhs.GetValue<double>());
            }
            else if(lhs.DataType == DType.Float32 || rhs.DataType == DType.Float32){
                return new Scalar<float>(lhs.GetValue<float>() + rhs.GetValue<float>());
            }
            else if(lhs.DataType == DType.Int64 || rhs.DataType == DType.Int64){
                return new Scalar<long>(lhs.GetValue<long>() + rhs.GetValue<long>());
            }
            else if(lhs.DataType == DType.Int32 || rhs.DataType == DType.Int32){
                return new Scalar<int>(lhs.GetValue<int>() + rhs.GetValue<int>());
            }
            else if(lhs.DataType == DType.Bool || rhs.DataType == DType.Bool){
                return new Scalar<bool>(lhs.GetValue<bool>() || rhs.GetValue<bool>());
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static Scalar operator -(Scalar lhs, Scalar rhs){
            if(lhs.DataType == DType.Float64 || rhs.DataType == DType.Float64){
                return new Scalar<double>(lhs.GetValue<double>() - rhs.GetValue<double>());
            }
            else if(lhs.DataType == DType.Float32 || rhs.DataType == DType.Float32){
                return new Scalar<float>(lhs.GetValue<float>() - rhs.GetValue<float>());
            }
            else if(lhs.DataType == DType.Int64 || rhs.DataType == DType.Int64){
                return new Scalar<long>(lhs.GetValue<long>() - rhs.GetValue<long>());
            }
            else if(lhs.DataType == DType.Int32 || rhs.DataType == DType.Int32){
                return new Scalar<int>(lhs.GetValue<int>() - rhs.GetValue<int>());
            }
            else if(lhs.DataType == DType.Bool || rhs.DataType == DType.Bool){
                throw new InvalidOperationException("Two bool scalar cannot execute sub operator.");
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static Scalar operator *(Scalar lhs, Scalar rhs){
            if(lhs.DataType == DType.Bool || rhs.DataType == DType.Bool){
                throw new InvalidOperationException("Bool scalar is not allowed for mul operation.");
            }
            else if(lhs.DataType == DType.Float64 || rhs.DataType == DType.Float64){
                return new Scalar<double>(lhs.GetValue<double>() - rhs.GetValue<double>());
            }
            else if(lhs.DataType == DType.Float32 || rhs.DataType == DType.Float32){
                return new Scalar<float>(lhs.GetValue<float>() - rhs.GetValue<float>());
            }
            else if(lhs.DataType == DType.Int64 || rhs.DataType == DType.Int64){
                return new Scalar<long>(lhs.GetValue<long>() - rhs.GetValue<long>());
            }
            else if(lhs.DataType == DType.Int32 || rhs.DataType == DType.Int32){
                return new Scalar<int>(lhs.GetValue<int>() - rhs.GetValue<int>());
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static Scalar operator /(Scalar lhs, Scalar rhs){
            if(lhs.DataType == DType.Bool || rhs.DataType == DType.Bool){
                throw new InvalidOperationException("Bool scalar is not allowed for div operation.");
            }
            else if(lhs.DataType == DType.Float64 || rhs.DataType == DType.Float64){
                return new Scalar<double>(lhs.GetValue<double>() / rhs.GetValue<double>());
            }
            else if(lhs.DataType == DType.Float32 || rhs.DataType == DType.Float32){
                return new Scalar<float>(lhs.GetValue<float>() / rhs.GetValue<float>());
            }
            else if(lhs.DataType == DType.Int64 || rhs.DataType == DType.Int64){
                return new Scalar<long>(lhs.GetValue<long>() / rhs.GetValue<long>());
            }
            else if(lhs.DataType == DType.Int32 || rhs.DataType == DType.Int32){
                return new Scalar<int>(lhs.GetValue<int>() / rhs.GetValue<int>());
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static Scalar operator %(Scalar lhs, Scalar rhs){
            if(lhs.DataType == DType.Bool || rhs.DataType == DType.Bool){
                throw new InvalidOperationException("Bool scalar is not allowed for mod operation.");
            }
            else if(lhs.DataType == DType.Float64 || rhs.DataType == DType.Float64){
                return new Scalar<double>(lhs.GetValue<double>() % rhs.GetValue<double>());
            }
            else if(lhs.DataType == DType.Float32 || rhs.DataType == DType.Float32){
                return new Scalar<float>(lhs.GetValue<float>() % rhs.GetValue<float>());
            }
            else if(lhs.DataType == DType.Int64 || rhs.DataType == DType.Int64){
                return new Scalar<long>(lhs.GetValue<long>() % rhs.GetValue<long>());
            }
            else if(lhs.DataType == DType.Int32 || rhs.DataType == DType.Int32){
                return new Scalar<int>(lhs.GetValue<int>() % rhs.GetValue<int>());
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static Scalar operator &(Scalar lhs, Scalar rhs){
            if(lhs.DataType != rhs.DataType){
                throw new MismatchedTypeException($"Only two scalar with same time could do & operation. " + 
                    "But the one is {Enum.GetName<DType>(lhs.DataType)}, the other is {Enum.GetName<DType>(rhs.DataType)}");
            }
            if(lhs.DataType == DType.Float64){
                throw new MismatchedTypeException($"Float64 is not allowed in & operation.");
            }
            else if(lhs.DataType == DType.Float32){
                throw new MismatchedTypeException($"Float32 is not allowed in & operation.");
            }
            else if(lhs.DataType == DType.Int64){
                return new Scalar<long>(lhs.GetValue<long>() & rhs.GetValue<long>());
            }
            else if(lhs.DataType == DType.Int32){
                return new Scalar<int>(lhs.GetValue<int>() & rhs.GetValue<int>());
            }
            else if(lhs.DataType == DType.Bool){
                return new Scalar<bool>(lhs.GetValue<bool>() & rhs.GetValue<bool>());
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static Scalar operator |(Scalar lhs, Scalar rhs){
            if(lhs.DataType != rhs.DataType){
                throw new MismatchedTypeException($"Only two scalar with same time could do | operation. " + 
                    "But the one is {Enum.GetName<DType>(lhs.DataType)}, the other is {Enum.GetName<DType>(rhs.DataType)}");
            }
            if(lhs.DataType == DType.Float64){
                throw new MismatchedTypeException($"Float64 is not allowed in | operation.");
            }
            else if(lhs.DataType == DType.Float32){
                throw new MismatchedTypeException($"Float32 is not allowed in | operation.");
            }
            else if(lhs.DataType == DType.Int64){
                return new Scalar<long>(lhs.GetValue<long>() | rhs.GetValue<long>());
            }
            else if(lhs.DataType == DType.Int32){
                return new Scalar<int>(lhs.GetValue<int>() | rhs.GetValue<int>());
            }
            else if(lhs.DataType == DType.Bool){
                return new Scalar<bool>(lhs.GetValue<bool>() | rhs.GetValue<bool>());
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static Scalar operator ^(Scalar lhs, Scalar rhs){
            if(lhs.DataType != rhs.DataType){
                throw new MismatchedTypeException($"Only two scalar with same time could do ^ operation. " + 
                    "But the one is {Enum.GetName<DType>(lhs.DataType)}, the other is {Enum.GetName<DType>(rhs.DataType)}");
            }
            if(lhs.DataType == DType.Float64){
                throw new MismatchedTypeException($"Float64 is not allowed in ^ operation.");
            }
            else if(lhs.DataType == DType.Float32){
                throw new MismatchedTypeException($"Float32 is not allowed in ^ operation.");
            }
            else if(lhs.DataType == DType.Int64){
                return new Scalar<long>(lhs.GetValue<long>() ^ rhs.GetValue<long>());
            }
            else if(lhs.DataType == DType.Int32){
                return new Scalar<int>(lhs.GetValue<int>() ^ rhs.GetValue<int>());
            }
            else if(lhs.DataType == DType.Bool){
                return new Scalar<bool>(lhs.GetValue<bool>() ^ rhs.GetValue<bool>());
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static bool operator ==(Scalar lhs, Scalar rhs){
            if(lhs.DataType == DType.Float64 || rhs.DataType == DType.Float64){
                return lhs.GetValue<double>() == rhs.GetValue<double>();
            }
            else if(lhs.DataType == DType.Float32 || rhs.DataType == DType.Float32){
                return lhs.GetValue<float>() == rhs.GetValue<float>();
            }
            else if(lhs.DataType == DType.Int64 || rhs.DataType == DType.Int64){
                return lhs.GetValue<long>() == rhs.GetValue<long>();
            }
            else if(lhs.DataType == DType.Int32 || rhs.DataType == DType.Int32){
                return lhs.GetValue<int>() == rhs.GetValue<int>();
            }
            else if(lhs.DataType == DType.Bool || rhs.DataType == DType.Bool){
                return lhs.GetValue<bool>() == rhs.GetValue<bool>();
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static bool operator !=(Scalar lhs, Scalar rhs){
            return !(lhs == rhs);
        }
        public static bool operator >(Scalar lhs, Scalar rhs){
            if(lhs.DataType == DType.Bool || rhs.DataType == DType.Bool){
                throw new MismatchedTypeException("Bool scalar is not allowed to do comparison.");
            }
            else if(lhs.DataType == DType.Float64 || rhs.DataType == DType.Float64){
                return lhs.GetValue<double>() > rhs.GetValue<double>();
            }
            else if(lhs.DataType == DType.Float32 || rhs.DataType == DType.Float32){
                return lhs.GetValue<float>() > rhs.GetValue<float>();
            }
            else if(lhs.DataType == DType.Int64 || rhs.DataType == DType.Int64){
                return lhs.GetValue<long>() > rhs.GetValue<long>();
            }
            else if(lhs.DataType == DType.Int32 || rhs.DataType == DType.Int32){
                return lhs.GetValue<int>() > rhs.GetValue<int>();
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static bool operator <(Scalar lhs, Scalar rhs){
            if(lhs.DataType == DType.Bool || rhs.DataType == DType.Bool){
                throw new MismatchedTypeException("Bool scalar is not allowed to do comparison.");
            }
            else if(lhs.DataType == DType.Float64 || rhs.DataType == DType.Float64){
                return lhs.GetValue<double>() < rhs.GetValue<double>();
            }
            else if(lhs.DataType == DType.Float32 || rhs.DataType == DType.Float32){
                return lhs.GetValue<float>() < rhs.GetValue<float>();
            }
            else if(lhs.DataType == DType.Int64 || rhs.DataType == DType.Int64){
                return lhs.GetValue<long>() < rhs.GetValue<long>();
            }
            else if(lhs.DataType == DType.Int32 || rhs.DataType == DType.Int32){
                return lhs.GetValue<int>() < rhs.GetValue<int>();
            }
            else{
                throw new NotImplementedException();
            }
        }
        public static bool operator >=(Scalar lhs, Scalar rhs){
            return !(lhs < rhs);
        }
        public static bool operator <=(Scalar lhs, Scalar rhs){
            return !(lhs > rhs);
        }
    }
    public class Scalar<T> : Scalar where T : struct{
        public Scalar(T value){
            Value = value;
            DataType = TensorTypeInfo.GetTypeInfo(typeof(T))._dtype;
        }
        public T Value{ get; set; }
        // public Scalar(int vaule, DType dataType) : base(new TensorMemory(1, dataType), new TensorLayout(new int[]{1}, dataType)){

        // }
        /// <summary>
        /// Get the value of the specified type. Since this function contains casting, 
        /// if you already have an instance of Scalar<T>, please use the Value property directly.
        /// </summary>
        /// <typeparam name="V"></typeparam>
        /// <returns></returns>
        /// <exception cref="NotImplementedException"></exception>
        public override V GetValue<V>() where V : struct{
            if(typeof(V) == typeof(bool)){
                return (V)(Convert.ToBoolean(Value) as ValueType);
            }
            else if(typeof(V) == typeof(double)){
                return (V)(Convert.ToDouble(Value) as ValueType);
            }
            else if(typeof(V) == typeof(float)){
                return (V)(Convert.ToSingle(Value) as ValueType);
            }
            else if(typeof(V) == typeof(long)){
                return (V)(Convert.ToInt64(Value) as ValueType);
            }
            else if(typeof(V) == typeof(int)){
                return (V)(Convert.ToInt32(Value) as ValueType);
            }
            else{
                throw new NotImplementedException();
            }
        }
        /// <summary>
        /// Get the value of the specified type. Since this function contains casting, 
        /// if you already have an instance of Scalar<T>, please use the Value property directly.
        /// </summary>
        /// <typeparam name="V"></typeparam>
        /// <param name="value"></param>
        public override void SetValue<V>(V value){
            if(value is null){
                throw new InvalidArgumentException("Null value is not allowed to be set to scalar.");
            }
            if(typeof(T) == typeof(double)){
                Value = (T)(Convert.ToDouble(value) as ValueType);
            }
            else if(typeof(T) == typeof(float)){
                Value = (T)(Convert.ToSingle(value) as ValueType);
            }
            else if(typeof(T) == typeof(long)){
                Value = (T)(Convert.ToInt64(value) as ValueType);
            }
            else if(typeof(T) == typeof(int)){
                Value = (T)(Convert.ToInt32(value) as ValueType);
            }
            else if(typeof(T) == typeof(Boolean)){
                Value = (T)(Convert.ToBoolean(value) as ValueType);
            }
            else{
                throw new NotImplementedException();
            }
        }
    }
}