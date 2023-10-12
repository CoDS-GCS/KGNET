package stardog_sample_udf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

import org.json.JSONArray;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonElement;
import com.google.gson.reflect.TypeToken;
import com.google.gson.Gson;
public class test {

	public static void main(String[] args) {
		String json="{\"https://dblp.org/rec/conf/ctrsa/Bultel22\":[\"CT-RSA\",\"zzzz\"],\"https://dblp.org/rec/conf/ctrsa/CaoSCCFW22\":[\"CT-RSA\",\"zz\"],\"https://dblp.org/rec/conf/ctrsa/Chevallier-Mames22\":\"CT-RSA\",\"https://dblp.org/rec/conf/ctrsa/CidIR22\":\"CT-RSA\",\"https://dblp.org/rec/conf/ctrsa/CuiHWW22\":\"CT-RSA\",\"https://dblp.org/rec/conf/ctrsa/FouotsaP22\":\"CT-RSA\",\"https://dblp.org/rec/conf/ctrsa/WuX22\":\"CT-RSA\",\"https://dblp.org/rec/conf/padl/2022\":\"Lecture Notes in Computer Science\",\"Inference_Times\":{\"subgraph_generation_time\":3e-06,\"transformation_time\":0.0,\"model_download_time\":13.107296,\"inference_time\":6.230733}}";
		json=json.replace("\"","'");
		System.out.println(json);
		JsonObject convertedObject  = new Gson().fromJson(json,JsonObject.class);
		String listString="";
		try
		{
//			listString=convertedObject.get("https://dblp.org/rec/conf/ctrsa/Chevallier-Mames22").getAsString();
			listString=convertedObject.get("https://dblp.org/rec/conf/ctrsa/CaoSCCFW22").getAsString();
			
			
		}
		catch(Exception ex)
		{
//			JsonArray list=convertedObject.getAsJsonArray("https://dblp.org/rec/conf/ctrsa/CaoSCCFW22");
//			listString="[";
//			for (JsonElement s : list)
//			{listString += s.toString() + ",";}		
			listString+=convertedObject.get("https://dblp.org/rec/conf/ctrsa/CaoSCCFW22").toString();		
		}
		System.out.println(listString);
		
	}

}
