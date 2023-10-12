package stardog_sample_udf;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.HashMap;
import java.util.Map;

import org.apache.http.client.methods.CloseableHttpResponse;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.DefaultHttpClient;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.util.EntityUtils;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
public class HTTP_Request {

	public static String  dopost_jsonString(String url,  JSONObject json)  throws IOException, InterruptedException{
    	CloseableHttpClient client = HttpClientBuilder.create().build();
    	HttpPost httpPost = new HttpPost(url);
    	String responseJSON=null;

    	        try {
    	            StringEntity entity = new StringEntity(json.toString());
    	            httpPost.setEntity(entity);

    	            // set your POST request headers to accept json contents
    	            httpPost.setHeader("Accept", "application/json");
    	            httpPost.setHeader("Content-type", "application/json");

    	            try {
    	                // your closeablehttp response
    	                CloseableHttpResponse response = client.execute(httpPost);

    	                // print your status code from the response
    	                System.out.println(response.getStatusLine().getStatusCode());

    	                // take the response body as a json formatted string 
    	                responseJSON = EntityUtils.toString(response.getEntity());

    	            } catch (IOException e) {
    	                e.printStackTrace();
    	            }
    	        }catch (Exception e) { }
       return responseJSON;
    }
	public static Map<String, Object> dopost(String url,  JSONObject json)  throws IOException, InterruptedException{
    	CloseableHttpClient client = HttpClientBuilder.create().build();
    	HttpPost httpPost = new HttpPost(url);
    	Map<String, Object> mapObj=null;

    	        try {
    	            StringEntity entity = new StringEntity(json.toString());
    	            httpPost.setEntity(entity);

    	            // set your POST request headers to accept json contents
    	            httpPost.setHeader("Accept", "application/json");
    	            httpPost.setHeader("Content-type", "application/json");

    	            try {
    	                // your closeablehttp response
    	                CloseableHttpResponse response = client.execute(httpPost);

    	                // print your status code from the response
    	                System.out.println(response.getStatusLine().getStatusCode());

    	                // take the response body as a json formatted string 
    	                String responseJSON = EntityUtils.toString(response.getEntity());
    	                mapObj = new Gson().fromJson(
    	                		responseJSON, new TypeToken<HashMap<String, Object>>() {}.getType());

//    	                // convert/parse the json formatted string to a json object
//    	                JSONObject jobj = new JSONObject(responseJSON);
//
//    	                //print your response body that formatted into json
//    	                System.out.println(jobj);

    	            } catch (IOException e) {
    	                e.printStackTrace();
    	            }
    	        }catch (Exception e) { }
       return mapObj;
    }
    public static void main(String[] args) throws IOException, InterruptedException, JSONException{
    	String JSON_STR="{"
    			+ "     \"model_id\": 92,"
    			+ "    	\"named_graph_uri\": \"https://dblp2022.org\","
    			+ "    	\"sparqlEndpointURL\": \"http://206.12.98.118:8890/sparql\","
    			+ "    	\"dataQuery\" : [\"PREFIX dblp2022: <https://dblp.org/rdf/schema#> PREFIX kgnet: <http://kgnet/> SELECT ?s ?p ?o from <https://dblp2022.org>   where { ?s ?p ?o { SELECT ?Publication as ?s  from <https://dblp2022.org>     WHERE {  ?Publication <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication . ?Publication dblp2022:publishedIn ?Org_Venue . ?Publication dblp2022:title ?Title . }    LIMIT 10  }  filter(!isBlank(?o)). } \"],"
    			+ "    	\"targetNodesQuery\": \"PREFIX dblp2022: <https://dblp.org/rdf/schema#> PREFIX kgnet: <http://kgnet/> SELECT distinct ?Publication as ?s  from <https://dblp2022.org>  WHERE {  ?Publication <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication . ?Publication dblp2022:publishedIn ?Org_Venue . ?Publication dblp2022:title ?Title . }   LIMIT 10\","
    			+ "    	\"topk\":2"
    			+ "}";
    	String url="http://206.12.99.65:64647/gml_inference/mid/92";
    	JSONObject json = new JSONObject(JSON_STR);
    	
//    	JSONObject json = new JSONObject();
//    	json.put("model_id", 92);
//    	json.put("named_graph_uri", "https://dblp2022.org");
//    	json.put("sparqlEndpointURL", "http://206.12.98.118:8890/sparql");
//    	JSONArray array = new JSONArray();
//    	array.put("PREFIX dblp2022: <https://dblp.org/rdf/schema#> PREFIX kgnet: <http://kgnet/> SELECT ?s ?p ?o from <https://dblp2022.org>   where { ?s ?p ?o { SELECT ?Publication as ?s  from <https://dblp2022.org>     WHERE {  ?Publication <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication . ?Publication dblp2022:publishedIn ?Org_Venue . ?Publication dblp2022:title ?Title . }    LIMIT 10  }  filter(!isBlank(?o)). } ");    	
//    	json.put("dataQuery", array);    	
//    	json.put("targetNodesQuery", "PREFIX dblp2022: <https://dblp.org/rdf/schema#> PREFIX kgnet: <http://kgnet/> SELECT distinct ?Publication as ?s  from <https://dblp2022.org>  WHERE {  ?Publication <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> dblp2022:Publication . ?Publication dblp2022:publishedIn ?Org_Venue . ?Publication dblp2022:title ?Title . }   LIMIT 10");
//    	json.put("topk",2);    	 
    	Map<String, Object> res=HTTP_Request.dopost(url,json);
    	System.out.println(res);


	}
}
