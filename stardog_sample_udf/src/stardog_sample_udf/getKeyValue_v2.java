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
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import com.complexible.stardog.plan.filter.Expression;
import com.complexible.stardog.plan.filter.functions.AbstractFunction;
import com.complexible.stardog.plan.filter.functions.string.StringFunction;
import com.stardog.stark.Values;
import com.google.gson.JsonObject;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
public final class getKeyValue_v2 extends AbstractFunction implements StringFunction {
	  public getKeyValue_v2 () {
	        super(2, new String[] { "getKeyValue_v2", "kgnet.getKeyValue_v2", "tag:stardog:api:kgnet:getKeyValue_v2" });
	    }
	    
	    public getKeyValue_v2 (final Expression theLeft, final Expression theRight) {
	        this();
	        this.mArgs = Lists.newArrayList(new Expression[]{ theLeft, theRight });
	    }
	    
	    private getKeyValue_v2 (final getKeyValue_v2 theExpr) {
	        super((AbstractFunction)theExpr);
	    }
	    
	    protected ValueOrError internalEvaluate(final Value... theArgs) {
	    	System.out.println("getKeyValue_v2 Called");
	    	System.out.println("Args[1]="+(theArgs[1]));
//	    	System.out.print(theArgs[0]);
//	    	System.out.println(theArgs[1]);
//	        if (!assertStringLiteral(theArgs[0]) || !assertStringLiteral(theArgs[1])) {
	        if (!assertStringLiteral(theArgs[1]) || !assertIRI(theArgs[0])) {
	            return (ValueOrError)ValueOrError.Error;
	        }
	        final String key = theArgs[0].toString();
	        final Literal Json_dict = (Literal)theArgs[1];
//	        final Literal key = (Literal)theArgs[1];
//	        if (!EvalUtil.isCompatible(Json_dict, Json_dict)) {
//	            return (ValueOrError)ValueOrError.Error;
//	        }
//	        Map<String, Object> mapObj = new Gson().fromJson(Json_dict.label(), new TypeToken<HashMap<String, String>>() {}.getType());
	        String json=Json_dict.label().replace("\"","'");
			System.out.println("json_dict="+json);
			JsonObject convertedObject  = new Gson().fromJson(json,JsonObject.class);
			String res="None";
			if (convertedObject.get(key)!=null)
			{
				try
				{
					res=convertedObject.get(key).getAsString();
				}
				catch(Exception ex)
				{
//					JsonArray list=convertedObject.getAsJsonArray("https://dblp.org/rec/conf/ctrsa/CaoSCCFW22");
//					listString="[";
//					for (JsonElement s : list)
//					{listString += s.toString() + ",";}		
					res=convertedObject.get(key).toString();		
				}
			}
//	        String res=(String)mapObj.get(key.label());
//	        return (ValueOrError)ValueOrError.Boolean.of(param1.label().endsWith(param2.label()));
	        return ValueOrError.General.of((Value)Values.literal(res, Json_dict.datatype()));
	    }
	    @Override
	    public void accept(final ExpressionVisitor theVisitor) {
	        theVisitor.visit((StringFunction)this);
	    }
	    @Override
	    public getKeyValue_v2  copy() {
	        return new getKeyValue_v2 (this);
	    }
	    public static void main(String[] args) {
			System.out.print("Call To getKeyValue_v2");
//			ToLowerCase tlc=new ToLowerCase();
//			com.complexible.common.rdf.model.ArrayLiteral v=new com.complexible.common.rdf.model.ArrayLiteral(10000);
//			tlc.internalEvaluate(v);		

		}
}
