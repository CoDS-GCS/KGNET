package stardog_sample_udf;
//
//Decompiled by Procyon v0.5.36
//

import com.complexible.common.base.Copyable;
import com.complexible.stardog.plan.filter.functions.Function;
import com.complexible.stardog.plan.filter.ExpressionVisitor;
import com.complexible.stardog.plan.filter.EvalUtil;
import com.stardog.stark.Literal;
import com.complexible.stardog.plan.filter.expr.ValueOrError;
import com.stardog.stark.Value;
import com.google.common.collect.Lists;
import com.complexible.stardog.plan.filter.Expression;
import com.complexible.stardog.plan.filter.functions.AbstractFunction;
import com.complexible.stardog.plan.filter.functions.string.StringFunction;
import com.stardog.stark.Values;
import java.util.List;
public final class Param2UDF extends AbstractFunction implements StringFunction {
	  public Param2UDF () {
	        super(2, new String[] { "Param2UDF", "kgnet.Param2UDF", "tag:stardog:api:kgnet:Param2UDF" });
	    }
	    
	    public Param2UDF (final Expression theLeft, final Expression theRight) {
	        this();
	        this.mArgs = Lists.newArrayList(new Expression[]{ theLeft, theRight });
	    }
	    
	    private Param2UDF (final Param2UDF theExpr) {
	        super((AbstractFunction)theExpr);
	    }
	    
	    protected ValueOrError internalEvaluate(final Value... theArgs) {
	    	System.out.println("Param2UDF Called");
	    	System.out.print("theArgs=");
	    	System.out.print(theArgs[0]);
	    	System.out.println(theArgs[1]);
	        if (!assertStringLiteral(theArgs[0]) || !assertStringLiteral(theArgs[1])) {
	            return (ValueOrError)ValueOrError.Error;
	        }
	        final Literal param1 = (Literal)theArgs[0];
	        final Literal param2 = (Literal)theArgs[1];
	        if (!EvalUtil.isCompatible(param1, param2)) {
	            return (ValueOrError)ValueOrError.Error;
	        }
	        String res=param1.label()+"->"+param2.label();
//	        return (ValueOrError)ValueOrError.Boolean.of(param1.label().endsWith(param2.label()));
	        return ValueOrError.General.of((Value)Values.literal(res, param1.datatype()));
	    }
	    @Override
	    public void accept(final ExpressionVisitor theVisitor) {
	        theVisitor.visit((StringFunction)this);
	    }
	    @Override
	    public Param2UDF  copy() {
	        return new Param2UDF (this);
	    }
	    public static void main(String[] args) {
			System.out.print("Call To ToLowerCase");
//			ToLowerCase tlc=new ToLowerCase();
//			com.complexible.common.rdf.model.ArrayLiteral v=new com.complexible.common.rdf.model.ArrayLiteral(10000);
//			tlc.internalEvaluate(v);		

		}
}
