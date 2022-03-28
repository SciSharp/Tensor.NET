using System.Text;

class InvalidLayoutException:Exception{
    private string _message;
    public override string Message => base.Message;
    // public InvalidLayoutException(TensorLayout layout){
    //     _message = $"{layout.ToString()} is an invalid TensorLayout.";
    // }
    public InvalidLayoutException(Span<int> shape){
        _message = $"Shape is an invalid TensorLayout.";
    }
}

class UnsopportedTypeException:Exception{
    private string _message;
    public override string Message => base.Message;
    public UnsopportedTypeException(){
        _message = $"Unsopprted Type";
    }
}