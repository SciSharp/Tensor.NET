using CodeGenerator;

var AddGenerator = new BasicArithGenerator() { Name = "Add", Expr = "x + y" };
CodeWriter.Write("Add", AddGenerator.GetCode());