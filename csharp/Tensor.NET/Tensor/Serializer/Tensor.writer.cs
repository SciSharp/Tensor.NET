using System.IO;
using System.Text;
using Tensornet.Common;
using Tensornet.Native;
using Tensornet.Exceptions;

namespace Tensornet{
    public enum TensorSerializationMode
    {
        Auto = 0,
        TensorNET,
        Numpy
    }
    public static class TensorWriter{
        private static byte[] _magicNumber;
        private static byte[] _versionNumber;

        static TensorWriter(){
            _magicNumber = Encoding.ASCII.GetBytes("Tensor.NET");
            _versionNumber = Encoding.ASCII.GetBytes(Tensor.VersionNumber);
        }

        public static void Write<T>(string path, Tensor<T> tensor, TensorSerializationMode mode = TensorSerializationMode.TensorNET) where T : struct, IEquatable<T>, IConvertible{
            using(var fs = new FileStream(path, FileMode.OpenOrCreate, FileAccess.Write)){
                WriteHeader(fs, tensor.TLayout);
                if(tensor.TLayout.IsContiguous()){
                    WriteContiguousData(fs, tensor.TMemory);
                }
                else{
                    WriteStrideData(fs, tensor);
                }
            }
        }

        public static async void WriteAsync<T>(string path, Tensor<T> tensor) where T : struct, IEquatable<T>, IConvertible{
            throw new NotImplementedException();
        }

        private static void WriteMagicAndVersion(FileStream fs){
            fs.Write(_magicNumber);
            fs.Write(_versionNumber);
        }

        private static void WriteHeader(FileStream fs, TensorLayout layout){
            WriteMagicAndVersion(fs);
            fs.Write(BitConverter.GetBytes(((int)layout.DType)));
            fs.Write(BitConverter.GetBytes(layout.NDim));
            for (int i = 0; i < TensorLayout.MAX_NDIM; i++)
            {
                fs.Write(BitConverter.GetBytes(layout.Shape[i]));
            }
        }

        private static void WriteContiguousData<T>(FileStream fs, TensorMemory<T> memory) where T : struct, IEquatable<T>, IConvertible{
            fs.Write(memory.AsByteSpan());
        }

        private static void WriteStrideData<T>(FileStream fs, Tensor<T> tensor) where T : struct, IEquatable<T>, IConvertible{
            byte[] data = new byte[TensorTypeInfo.GetTypeSize(tensor.TLayout.DType)];
            Span<byte> dataSpan = new Span<byte>(data);
            foreach(T item in tensor){
                bool actionRes = tensor.TLayout.DType switch
                {
                    DType.Int32 => BitConverter.TryWriteBytes(dataSpan, Convert.ToInt32(item)),
                    DType.Int64 => BitConverter.TryWriteBytes(dataSpan, Convert.ToInt64(item)),
                    DType.Float32 => BitConverter.TryWriteBytes(dataSpan, Convert.ToSingle(item)),
                    DType.Float64 => BitConverter.TryWriteBytes(dataSpan, Convert.ToDouble(item)),
                    DType.Bool => BitConverter.TryWriteBytes(dataSpan, Convert.ToBoolean(item)),
                    _ => false
                };
                if(!actionRes){
                    throw new NNRuntimeException("Failed to write byte to target span.");
                }
                fs.Write(dataSpan);
            }
        }
    }

    public partial class Tensor<T>{
        public void Save(string path, TensorSerializationMode mode = TensorSerializationMode.TensorNET){
            TensorWriter.Write(path, this, mode);
        }
    }
}