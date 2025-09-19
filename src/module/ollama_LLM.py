# from typing import Any
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage
# import numpy as np
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
# import torch
# import warnings
# import os


class OLLAMA_LLM:
    def __init__(self, model_name, port, max_token, seed):
        self.model_name = model_name
        self.port = port
        self.chat_model = ChatOllama(model=model_name, 
                                base_url=f"http://localhost:{port}",
                                num_predict=max_token,
                                seed=seed)
        self.call_cnt = 0
        self.token_cnt = 0

    def generate_chat_response(self, messages):
        langchain_messages = []
        for message in messages:
            if "role" in message.keys():
                if message['role'] == 'user':
                    langchain_messages.append(HumanMessage(content=message['content']))
                elif message['role'] == 'assistant':
                    langchain_messages.append(AIMessage(content=message['content']))
                elif message['role'] == 'system':
                    langchain_messages.append(SystemMessage(content=message['content']))
        
        message = self.chat_model.invoke(langchain_messages)
        self.call_cnt += 1
        if message.usage_metadata is not None:
            self.token_cnt += message.usage_metadata["output_tokens"]
        else:
            print(message)
        # print(f"cumul_call_cnt: {self.call_cnt}, cumul_token_cnt: {self.token_cnt}")
        return message
    
    def convert_to_dict(self, message):
        return_dict = dict()
        # return_dict["role"] = message.role
        return_dict['content'] = message.content
        return return_dict

    def get_call_cnt(self):
        return self.call_cnt
    
    def get_token_cnt(self):
        return self.token_cnt
    

# class OLLAMA_Embedding_LLM:
#     def __init__(self, model_name, port):
#         self.model_name = model_name
#         self.port = port
#         self.ollama_embed = OllamaEmbeddings(model=model_name, 
#                                              base_url=f"http://localhost:{port}")
        
#     def embed(self, query):
#         embed = self.ollama_embed.embed_query(query)
#         embed = np.array(embed)
#         return embed

# class Hidden_State_LLM:
#     def __init__(self, model_name):
#         if model_id == "qwen2:0.5b":
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
#                 torch_dtype="auto",
#                 device_map="auto", cache_dir="./ckpts/llm"
#             )
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4", cache_dir="./ckpts/llm")
                
#         elif model_id == "gemma2:2b":
#             self.model = gemma2_2b()
#         elif model_id == "llama3.1:8b":
#             self.model = llama3_1_8b()
#         elif model_id == "phi3.5-mini":
#             self.model = phi3_5_mini()
#         else:
#             raise ValueError(f"Model {model_id} not found.")


# class qwen2_0_5b():
#     def __init__(self):
#         

#     def generate(self, message, max_tokens):

#         text = self.tokenizer.apply_chat_template(
#             message,
#             tokenize=False,
#             add_generation_prompt=True
#         )

#         model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

#         generated_ids = self.model.generate(
#             model_inputs.input_ids,
#             max_new_tokens=512
#         )

#         generated_ids = [
#             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#         ]

#         response = self.tokenizer.batch_decode(
#             generated_ids, skip_special_tokens=True)[0]

#         return response


# # class phi3_5_mini():
# #     def __init__(self):
# #         model = AutoModelForCausalLM.from_pretrained(
# #             "anakin87/Phi-3.5-mini-ITA",
# #             device_map="auto",
# #             torch_dtype=torch.bfloat16,
# #             trust_remote_code=True,
# #         )
# #         tokenizer = AutoTokenizer.from_pretrained(
# #             "anakin87/Phi-3.5-mini-ITA", trust_remote_code=True)
# #         self.pipe = pipeline(
# #             "text-generation", model=model, tokenizer=tokenizer)

# #     def generate(self, message, max_tokens):

# #         # prompt = "Large language models have been used in a variety of applications, including chatbots, recommendation engines, and online gaming."

# #         # user_input = "Puoi spiegarmi brevemente la differenza tra imperfetto e passato prossimo in italiano e quando si usano?"
# #         # messages = [{"role": "user", "content": user_input}]
# #         outputs = self.pipe(message, max_new_tokens=500,
# #                             do_sample=True, temperature=0.001)
# #         return outputs[0]["generated_text"][2]

# # elif model_id == "gemma2:2b":
# #     quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

# #     tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
# #     model = AutoModelForCausalLM.from_pretrained(
# #         "google/gemma-2-2b",
# #         low_cpu_mem_usage=True,
# #         quantization_config=quantization_config,
# #     )

# #     input_text = "Write me a poem about Machine Learning."
# #     input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# #     outputs = model.generate(**input_ids, max_new_tokens=200)
# #     print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# # elif model_id == "llama3.1:8b":
# #     model_id = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

# #     model = AutoModelForCausalLM.from_pretrained(
# #         model_id,
# #         low_cpu_mem_usage=True
# #     )
# #     tokenizer = AutoTokenizer.from_pretrained(model_id)

# #     input_text = "Write me a poem about Machine Learning."
# #     input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# #     outputs = model.generate(**input_ids, max_new_tokens=200)
# #     print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# message = [{"role": "user", "content": "Give me a short introduction to large language model."}]

# message = [{'role': 'user', 
#             'content': 'The candidate relations: "royalty.kingdom.rulers", "sports.sports_team_location.teams", "base.dspl.world_bank.wdi_gdf.country", "film.film_location.featured_in_films", "symbols.flag_referent.flag", "olympics.olympic_participating_country.athletes", "olympics.olympic_participating_country.ioc_code", "olympics.olympic_participating_country.medals_won", "olympics.olympic_participating_country.olympics_participated_in", "kg.object_profile.prominent_type", "base.ontologies.ontology_instance.equivalent_instances", "award.award_presenting_organization.awards_presented", "base.ranker.rankerurlname", "food.beer_country_region.beers_from_here", "base.aareas.schema.administrative_area.adjectival_form", "base.aareas.schema.administrative_area.administrative_area_type", "base.aareas.schema.administrative_area.administrative_children", "base.aareas.schema.administrative_area.administrative_parent", "base.aareas.schema.administrative_area.pertinent_type", "base.aareas.schema.administrative_area.short_name", "base.aareas.schema.administrative_area.subdividing_type", "base.athletics.athletics_country.championships_athletes_performances", "organization.organization_member.member_of", "freebase.linguistic_hint.adjectival_form", "meteorology.cyclone_affected_area.cyclones", "organization.organization_scope.organizations_with_this_scope", "topic_server.population_number", "location.country.administrative_divisions", "location.country.calling_code", "location.country.capital", "location.country.currency_used", "location.country.fifa_code", "location.country.fips10_4", "location.country.first_level_divisions", "location.country.form_of_government", "location.country.internet_tld", "location.country.iso3166_1_alpha2", "location.country.iso3166_1_shortname", "location.country.iso_alpha_3", "location.country.iso_numeric", "location.country.languages_spoken", "location.country.national_anthem", "location.country.official_language", "en", "location.dated_location.date_founded", "authority.daylife.topic", "authority.fifa", "travel.travel_destination.tourist_attractions", "freebase.valuenotation.is_reviewed", "book.book_subject.works", "location.location.adjectival_form", "location.location.area", "location.location.containedby", "location.location.contains", "authority.iso.3166-1.numeric", "authority.iso.3166-1.alpha-2", "location.location.events", "location.location.geolocation", "authority.iso.3166-1.alpha-3", "location.location.nearby_airports", "location.location.people_born_here", "location.location.time_zones", "base.biblioness.bibs_location.loc_type", "location.statistical_region.agriculture_as_percent_of_gdp", "location.statistical_region.brain_drain_percent", "location.statistical_region.child_labor_percent", "location.statistical_region.co2_emissions_per_capita", "location.statistical_region.consumer_price_index", "location.statistical_region.cpi_inflation_rate", "location.statistical_region.debt_service_as_percent_of_trade_volume", "location.statistical_region.deposit_interest_rate", "location.statistical_region.electricity_consumption_per_capita", "location.statistical_region.diesel_price_liter", "location.statistical_region.energy_use_per_capita", "location.statistical_region.external_debt_stock", "location.statistical_region.fertility_rate", "location.statistical_region.foreign_direct_investment_net_inflows", "location.statistical_region.gdp_growth_rate", "location.statistical_region.gdp_nominal", "location.statistical_region.gdp_nominal_per_capita", "location.statistical_region.gdp_real", "location.statistical_region.gender_balance_members_of_parliament", "location.statistical_region.gni_in_ppp_dollars", "fictional_universe.fictional_setting.fictional_characters_born_here", "location.statistical_region.gni_per_capita_in_ppp_dollars", "location.statistical_region.gross_savings_as_percent_of_gdp", "location.statistical_region.health_expenditure_as_percent_of_gdp", "location.statistical_region.high_tech_as_percent_of_manufactured_exports", "location.statistical_region.internet_users_percent_population", "location.statistical_region.labor_participation_rate", "base.locations.countries.continent", "location.statistical_region.life_expectancy", "location.statistical_region.literacy_rate", "sports.sport_country.athletic_performances", "location.statistical_region.long_term_unemployment_rate", "sports.sport_country.athletes", "location.statistical_region.market_cap_of_listed_companies_as_percent_of_gdp", "location.statistical_region.merchandise_trade_percent_of_gdp", "location.statistical_region.military_expenditure_percent_gdp", "sports.sport_country.multi_event_tournaments_participated_in", "location.statistical_region.net_migration", "location.statistical_region.official_development_assistance", "location.statistical_region.part_time_employment_percent", "base.popstra.location.vacationers", "location.statistical_region.population", "location.statistical_region.population_growth_rate", "location.statistical_region.poverty_rate_2dollars_per_day", "location.statistical_region.prevalence_of_undernourisment", "government.governmental_jurisdiction.governing_officials", "government.governmental_jurisdiction.agencies", "government.governmental_jurisdiction.government_bodies", "location.statistical_region.renewable_freshwater_per_capita", "location.statistical_region.size_of_armed_forces", "location.statistical_region.time_required_to_start_a_business", "location.statistical_region.trade_balance_as_percent_of_gdp".\nThe question is "what does jamaican people speak?" and you\'ll start with "jamaican". To answer this question, typically you would need to identify some relations that correspond to the meaning of the question. Therefore, select one relation from the candidate relations above that can be used to answer the question. Provide only one relevant relation that\'s present in the candidates, and begin your response with "The relevant relation: ".'}]

# max_tokens = 30
# model = get_LLM("qwen2:0.5b")
# print(model.generate(message, max_tokens))

    

if __name__ == "__main__":
    model = OLLAMA_LLM(model_name="", port=11500, max_token=100, seed=0)
    message_input = [{'role': 'user', 'content': 'hello'},]
    response = model.generate_chat_response(message_input)
    print(response)


