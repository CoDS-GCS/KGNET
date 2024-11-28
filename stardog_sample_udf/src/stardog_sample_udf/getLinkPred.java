package stardog_sample_udf;
//g
//Decompiled by Procyon v0.5.36
//

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.HashMap;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.complexible.stardog.plan.filter.functions.ReturnsString;
import com.complexible.common.base.Copyable;
import com.complexible.stardog.index.dictionary.MappingDictionary;
import java.util.stream.Stream;

import org.json.JSONObject;

import java.util.regex.PatternSyntaxException;
import com.stardog.stark.Value;
import com.complexible.common.rdf.model.ArrayLiteral;
import java.util.function.ToLongFunction;
import java.util.Objects;
import com.stardog.stark.Values;
import java.util.Arrays;
import com.stardog.stark.Literal;
import com.complexible.stardog.plan.filter.functions.AbstractFunction;
import com.complexible.stardog.plan.filter.expr.ValueOrError;
import com.complexible.stardog.plan.filter.ValueSolution;
import com.complexible.stardog.plan.filter.ExpressionVisitor;
import com.complexible.stardog.plan.filter.functions.Function;
import java.util.List;
import java.util.Map;

import com.google.common.collect.Lists;
import com.complexible.stardog.plan.filter.Expression;
import com.complexible.stardog.plan.filter.AbstractExpression;
import com.complexible.stardog.plan.filter.functions.string.StringFunction;

public class getLinkPred extends AbstractExpression implements StringFunction
{
 public getLinkPred() {
     super(new Expression[0]);
//     super(2, new String[] { "getPredDict", "kgnet.getPredDict", "tag:stardog:api:kgnet:getPredDict" });
 }
 
 public getLinkPred(final Expression theLeft, final Expression theRight) {
     this();
     this.mArgs = Lists.newArrayList(new Expression[] { theLeft, theRight });
 }
 
 private getLinkPred(final getLinkPred theExpr) {
     super((AbstractExpression)theExpr);
 }
 
 public String getName() {
     return "tag:stardog:api:kgnet:getLinkPred";
 }
 
 public List<String> getNames() {
     return (List<String>)Lists.newArrayList(new String[] { this.getName() });
 }
 
 public Function copy() {
     return (Function)new getLinkPred(this);
 }
 
 public void accept(final ExpressionVisitor theVisitor) {
     theVisitor.visit((StringFunction)this);
 }
 
 public ValueOrError evaluate(final ValueSolution theValueSolution) {
	 System.out.print("getLinkPred Called");
     final List<Expression> args = (List<Expression>)this.getArgs();
     if (args.size() < 2 || args.size() > 3) {
         return (ValueOrError)ValueOrError.Error;
     }
     final ValueOrError arg0 = args.get(0).evaluate(theValueSolution);
     final ValueOrError arg2 = args.get(1).evaluate(theValueSolution);
     if (arg0.isError() || arg2.isError()) {
         return (ValueOrError)ValueOrError.Error;
     }
     if (!AbstractFunction.assertStringLiteral(arg0.value()) || !AbstractFunction.assertStringLiteral(arg2.value())) {
         return (ValueOrError)ValueOrError.Error;
     }
     final String model_url = ((Literal)arg0.value()).label();
     final String JSON_STR = ((Literal)arg2.value()).label();
     int limit = 0;
     if (args.size() == 3) {
         final ValueOrError arg3 = args.get(2).evaluate(theValueSolution);
         if (arg3.isError() || !AbstractFunction.assertIntegerLiteral(arg3.value())) {
             return (ValueOrError)ValueOrError.Error;
         }
         limit = Integer.valueOf(((Literal)arg3.value()).label());
     }
     try {
//         final String[] components = toSplit.split(splitBy, limit);
//         final Stream<Object> map = Arrays.stream(components).map((java.util.function.Function<? super String, ?>)Values::literal);
//         final MappingDictionary dictionary = theValueSolution.getDictionary();
//         Objects.requireNonNull(dictionary);
//         final long[] ids = map.mapToLong(elem ->dictionary.add((Value)elem)).toArray();
//         return ValueOrError.General.of((Value)new ArrayLiteral(ids));
//    	 String JSON_STR="{"
//     			+ "     \"model_id\": 92,"
//     			+ "    	\"named_graph_uri\": \"https://dblp2022.org\","
//     			+ "    	\"sparqlEndpointURL\": \"http://206.12.98.118:8890/sparql\","
//     			+ "    	\"dataQuery\" : [\"PREFIX dblp2022: <https://dblp.org/rdf/schema#> PREFIX kgnet: <http://kgnet/> SELECT ?s ?p ?o from <https://dblp2022.org>   where { ?s ?p ?o { SELECT ?Publication as ?s  from <https://dblp2022.org>     WHERE {  ?Publication <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication . ?Publication dblp2022:publishedIn ?Org_Venue . ?Publication dblp2022:title ?Title . }    LIMIT 10  }  filter(!isBlank(?o)). } \"],"
//     			+ "    	\"targetNodesQuery\": \"PREFIX dblp2022: <https://dblp.org/rdf/schema#> PREFIX kgnet: <http://kgnet/> SELECT distinct ?Publication as ?s  from <https://dblp2022.org>  WHERE {  ?Publication <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication . ?Publication dblp2022:publishedIn ?Org_Venue . ?Publication dblp2022:title ?Title . }   LIMIT 10\","
//     			+ "    	\"topk\":2"
//     			+ "}";
//     	String url="http://206.12.99.65:64647/gml_inference/mid/92";
     	JSONObject json = new JSONObject(JSON_STR);   
     	String res=HTTP_Request.dopost_jsonString(model_url ,json);
     	System.out.println("getLinkPred Res="+res);
     	return ValueOrError.General.of((Value)Values.literal(res));
     	
//        return ValueOrError.General.of((Value)new ArrayLiteral(res));
     	
     }
     catch (PatternSyntaxException e) {
         return (ValueOrError)ValueOrError.Error;
     }
     catch (Exception e) {
         return (ValueOrError)ValueOrError.Error;
     }
 }
 public static void main(String[] args) {
	 System.out.print("Call To getLinkPred");
//		ToLowerCase tlc=new ToLowerCase();
//		com.complexible.common.rdf.model.ArrayLiteral v=new com.complexible.common.rdf.model.ArrayLiteral(10000);
//		tlc.internalEvaluate(v);		

	}

}
