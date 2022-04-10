namespace Numnet.Exceptions{
    public class UnsupportedTypeException:Exception{
        protected string _message;
        public override string Message => _message;
        public UnsupportedTypeException(string message){
            _message = message;
        }
        public UnsupportedTypeException(){
            _message = "";
        }
    }
}