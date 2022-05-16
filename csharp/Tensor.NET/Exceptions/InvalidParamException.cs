namespace Tensornet.Exceptions{
    public class InvalidParamException:Exception{
        protected string _message;
        public override string Message => _message;
        public InvalidParamException(string message){
            _message = message;
        }
        public InvalidParamException(){
            _message = "";
        }
    }
}