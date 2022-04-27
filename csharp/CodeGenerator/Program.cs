using CodeGenerator;
using System.Collections.Generic;

var AddGenerator = new BasicArithGenerator() { Name = "Add", Expr = "x + y" };
CodeWriter.Write("Add", AddGenerator.GetCode());

var SubGenerator = new BasicArithGenerator() { Name = "Sub", Expr = "x - y" };
CodeWriter.Write("Sub", SubGenerator.GetCode());

var DivGenerator = new BasicArithGenerator() { Name = "Div", Expr = "x / y" };
CodeWriter.Write("Div", DivGenerator.GetCode());

var ModGenerator = new BasicArithGenerator() { Name = "Mod", Expr = "x % y" };
CodeWriter.Write("Mod", ModGenerator.GetCode());

var AbsGenerator = new BasicFunctionGenerator() { Name = "Abs", Expr = "System.Math.Abs(x)" };
CodeWriter.Write("Abs", AbsGenerator.GetCode());

var FloorGenerator = new BasicFunctionGenerator() { Name = "Floor", Expr = "System.Math.Floor(x)" };
CodeWriter.Write("Floor", FloorGenerator.GetCode());

var CeilGenerator = new BasicFunctionGenerator() { Name = "Ceil", Expr = "System.Math.Ceil(x)" };
CodeWriter.Write("Ceil", CeilGenerator.GetCode());

var RoundGenerator = new BasicFunctionGenerator() { Name = "Round", Expr = "System.Math.Round(x)" };
CodeWriter.Write("Round", RoundGenerator.GetCode());

var PowGenerator1 = new StaticUtilGenerator() { 
    Name = "Pow", 
    Parameters = new List<string>(new string[]{"int y"}), 
    Expr = "System.Math.Pow(x, y)" };
var PowGenerator2 = new ReversedStaticUtilGenerator() { 
    Name = "Pow", 
    Parameters = new List<string>(new string[]{"int y"}), 
    Expr = "System.Math.Pow(y, x)" };
CodeWriter.Write("Pow", PowGenerator1.GetCode() + PowGenerator2.GetCode());

var ClampGenerator = new StaticUtilGenerator(){
    Name = "Clamp", 
    Parameters = new List<string>(new string[]{"T min", "T max"}), 
    Expr = "System.Math.Clamp(x, min, max)" };
CodeWriter.Write("Clamp", ClampGenerator.GetCode());

var ExpGenerator = new StaticUtilGenerator() { 
    Name = "Exp", 
    Expr = "System.Math.Exp(x)" };
CodeWriter.Write("Exp", ExpGenerator.GetCode());

var LogGenerator1 = new StaticUtilGenerator() { 
    Name = "Log", 
    Parameters = new List<string>(new string[]{"double y"}), 
    Expr = "System.Math.Log(y, x)" };
var LogGenerator2 = new ReversedStaticUtilGenerator() { 
    Name = "Log", 
    Parameters = new List<string>(new string[]{"double baseValue"}), 
    Expr = "System.Math.Log(x, baseValue)" };
CodeWriter.Write("Log", LogGenerator1.GetCode() + LogGenerator2.GetCode());

var Log2Generator = new StaticUtilGenerator() { 
    Name = "Log2", 
    Expr = "System.Math.Log2(x)" };
CodeWriter.Write("Log2", Log2Generator.GetCode());

var Log10Generator = new StaticUtilGenerator() { 
    Name = "Log10", 
    Expr = "System.Math.Log10(x)" };
CodeWriter.Write("Log10", Log10Generator.GetCode());

var MaxGenerator = new BasicArithGenerator() { Name = "Max", Expr = "x > y ? x : y" };
CodeWriter.Write("Max", MaxGenerator.GetCode());

var MinGenerator = new BasicArithGenerator() { Name = "Min", Expr = "x > y ? y : x" };
CodeWriter.Write("Min", MinGenerator.GetCode());