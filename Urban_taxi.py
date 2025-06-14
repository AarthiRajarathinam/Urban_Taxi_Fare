##########################################################
#Oueries - Obesity Table
#########################################################


# 1. Top 5 regions with the highest average obesity levels in the most recent year(2022)

ob_Avg = df_obesity.groupby('Region')['Mean_Estimate'].mean().reset_index()
ob_Avg = ob_Avg.sort_values(by='Mean_Estimate', ascending=False).head(5)    
ob_Avg['Year'] = 2022  # Adding the year column
ob_Avg.rename(columns={'Mean_Estimate': 'Average_Obesity_Level'}, inplace=True)
ob_Avg = ob_Avg.reset_index(drop=True)
#st.write("Top 5 regions with the highest average obesity levels in 2022:")
#st.dataframe(ob_Avg)

#############################################################################

# 2. Top 5 countries with highest obesity estimates

ob_top_countries = df_obesity.sort_values(by='Mean_Estimate', ascending=False).head(5)
ob_top_countries = ob_top_countries[['Country', 'Mean_Estimate', 'Year']].reset_index(drop=True)
ob_top_countries.rename(columns={'Mean_Estimate': 'Obesity_Estimate'}, inplace=True)
#st.write("Top 5 countries with the highest obesity estimates:")
#st.dataframe(ob_top_countries)

#############################################################################

# 3. Obesity trend in India over the years(Mean_estimate)

ob_india_trend = df_obesity[df_obesity['Country'] == 'India'][['Year', 'Mean_Estimate']]
ob_india_trend = ob_india_trend.reset_index(drop=True)          
ob_india_trend.rename(columns={'Mean_Estimate': 'Obesity_Estimate'}, inplace=True)
#st.write("Obesity trend in India over the years:")             
#st.dataframe(ob_india_trend)

#############################################################################

# 4. Average obesity by gender 

#ob_avg_gender = df_obesity.groupby(['Gender', 'Year'])['Mean_Estimate'].mean().reset_index()

ob_avg_gender = df_obesity.groupby(['Gender'])['Mean_Estimate'].mean().reset_index()
ob_avg_gender.rename(columns={'Mean_Estimate': 'Average_Obesity_Level'}, inplace=True)
ob_avg_gender = ob_avg_gender.reset_index(drop=True)          
#st.write("Average obesity by gender:")
#st.dataframe(ob_avg_gender)                                                     

#############################################################################

# 5. Country count by obesity level category and age group

ob_country_count = df_obesity.groupby(['Obesity_Level', 'Age_Group']).size().reset_index(name='Country_Count')

ob_country_count = ob_country_count.reset_index(drop=True).drop_duplicates()      
#st.write("Country count by obesity level category and age group:")
#st.dataframe(ob_country_count)


###############################################################################

#6. Top 5 countries least reliable countries(with highest CI_Width) and Top 5 most consistent countries (smallest average CI_Width)

ob_least_reliable = df_obesity.sort_values(by='CI_Width', ascending=False).head(5)
ob_least_reliable = ob_least_reliable[['Country', 'CI_Width', 'Year']].reset_index(drop=True)               
ob_least_reliable.rename(columns={'CI_Width': 'Highest_CI_Width'}, inplace=True)
ob_most_consistent = df_obesity.groupby('Country')['CI_Width'].mean().reset_index() 
ob_most_consistent = ob_most_consistent.sort_values(by='CI_Width').head(5).reset_index(drop=True)
ob_most_consistent.rename(columns={'CI_Width': 'Lowest_Average_CI_Width'}, inplace=True)  
  
#st.write("Top 5 least reliable countries (highest CI_Width):")
#st.dataframe(ob_least_reliable)
#st.write("Top 5 most consistent countries (smallest average CI_Width):")
#st.dataframe(ob_most_consistent)    

##################################################################################

#7. Average obesity by age group

ob_avg_age_group = df_obesity.groupby('Age_Group')['Mean_Estimate'].mean().reset_index()
ob_avg_age_group.rename(columns={'Mean_Estimate': 'Average_Obesity_Level'}, inplace=True)           
ob_avg_age_group = ob_avg_age_group.reset_index(drop=True)
#st.write("Average obesity by age group:")
#st.dataframe(ob_avg_age_group)


#################################################################################

#8. Top 10 Countries with consistent low obesity (low average + low CI)over the years

ob_consistent_low = df_obesity.groupby('Country').agg({'Mean_Estimate': 'mean', 'CI_Width': 'mean'}).reset_index()
ob_consistent_low = ob_consistent_low.sort_values(by=['Mean_Estimate', 'CI_Width']).head(10)    
ob_consistent_low.rename(columns={'Mean_Estimate': 'Average_Obesity_Level', 'CI_Width': 'Average_CI_Width'}, inplace=True)
ob_consistent_low = ob_consistent_low.reset_index(drop=True)
#st.write("Top 10 countries with consistent low obesity (low average + low CI):")
#st.dataframe(ob_consistent_low)


####################################################################################

#9. Countries where female obesity exceeds male by large margin (same year)



######################################################################################

#10. Global average obesity percentage per year 

ob_global_avg = df_obesity.groupby('Year')['Mean_Estimate'].mean().reset_index()
ob_global_avg.rename(columns={'Mean_Estimate': 'Global_Average_Obesity_Percentage'}, inplace=True)              
ob_global_avg = ob_global_avg.reset_index(drop=True)
#st.write("Global average obesity percentage per year:")
#st.dataframe(ob_global_avg)



##########################################################
#Oueries - Malnutrition Table
#########################################################


# 1. Avg. malnutrition by age group

mal_avg_age = df_malnutrition.groupby('Age_Group')['Mean_Estimate'].mean().reset_index()
mal_avg_age.rename(columns={'Mean_Estimate': 'Average_Malnutrition_Level'}, inplace=True)          
mal_avg_age = mal_avg_age.reset_index(drop=True)
#st.write("Average malnutrition by age group:")
#st.dataframe(mal_avg_age) 

#############################################################################

# 2. Top 5 countries with highest malnutrition(mean_estimate)

mal_top_countries = df_malnutrition.sort_values(by='Mean_Estimate', ascending=False).head(5)
mal_top_countries = mal_top_countries[['Country', 'Mean_Estimate', 'Year']].reset_index(drop=True)      
mal_top_countries.rename(columns={'Mean_Estimate': 'Malnutrition_Estimate'}, inplace=True)
#st.write("Top 5 countries with the highest malnutrition estimates:")
#st.dataframe(mal_top_countries)



#############################################################################

# 3. Malnutrition trend in African region over the years

mal_africa_trend = df_malnutrition[df_malnutrition['Country'] == 'Africa'][['Year', 'Mean_Estimate']]
mal_africa_trend = mal_africa_trend.reset_index(drop=True)
mal_africa_trend.rename(columns={'Mean_Estimate': 'Malnutrition_Estimate'}, inplace=True)
#st.write("Malnutrition trend in Africa over the years:")
#st.dataframe(mal_africa_trend)

#############################################################################

# 4. Gender-based average malnutrition

mal_avg_gender = df_obesity.groupby(['Gender'])['Mean_Estimate'].mean().reset_index()
mal_avg_gender.rename(columns={'Mean_Estimate': 'Average_Obesity_Level'}, inplace=True)
mal_avg_gender = mal_avg_gender.reset_index(drop=True)          
#st.write("Average obesity by gender:")
#st.dataframe(mal_avg_gender)                                              

#############################################################################

# 5. Malnutrition level-wise (average CI_Width by age group)

mal_avg_ci_age = df_malnutrition.groupby(['Age_Group', 'CI_Width'])['Mean_Estimate'].mean().reset_index()
mal_avg_ci_age.rename(columns={'Mean_Estimate': 'Average_Malnutrition_Level'}, inplace=True)
mal_avg_ci_age = mal_avg_ci_age.reset_index(drop=True)          
#st.write("Average malnutrition level by age group and CI_Width:")
#st.dataframe(mal_avg_ci_age)

###############################################################################

#6. Yearly malnutrition change in specific countries(India, Nigeria, Brazil)

mal_yearly_change = df_malnutrition[df_malnutrition['Country'].isin(['India', 'Nigeria', 'Brazil'])][['Country', 'Year', 'Mean_Estimate']]
mal_yearly_change = mal_yearly_change.reset_index(drop=True)    
mal_yearly_change.rename(columns={'Mean_Estimate': 'Malnutrition_Estimate'}, inplace=True)
#st.write("Yearly malnutrition change in specific countries (India, Nigeria, Brazil):")
#st.dataframe(mal_yearly_change)


##################################################################################

# 7. Regions with lowest malnutrition averages

mal_low_regions = df_malnutrition.groupby('Region')['Mean_Estimate'].mean().reset_index()
mal_low_regions = mal_low_regions.sort_values(by='Mean_Estimate').head(5)       
mal_low_regions.rename(columns={'Mean_Estimate': 'Average_Malnutrition_Level'}, inplace=True)
mal_low_regions = mal_low_regions.reset_index(drop=True)
#st.write("Regions with lowest malnutrition averages:")
#st.dataframe(mal_low_regions)

#################################################################################

#8. Countries with increasing malnutrition 

mal_increasing_countries = df_malnutrition.groupby('Country').agg({'Mean_Estimate': ['min', 'max']}).reset_index()
mal_increasing_countries.columns = ['Country', 'Min_Malnutrition', 'Max_Malnutrition']
mal_increasing_countries['Difference'] = mal_increasing_countries['Max_Malnutrition'] - mal_increasing_countries['Min_Malnutrition']
mal_increasing_countries = mal_increasing_countries[mal_increasing_countries['Difference'] > 0]     
mal_increasing_countries = mal_increasing_countries.reset_index(drop=True)
#st.write("Countries with increasing malnutrition levels:")
#st.dataframe(mal_increasing_countries)

####################################################################################

#9. Min/Max malnutrition levels year-wise comparison
mal_min_max_yearly = df_malnutrition.groupby('Year')['Mean_Estimate'].agg(['min', 'max']).reset_index()
mal_min_max_yearly.rename(columns={'min': 'Min_Malnutrition', 'max': 'Max_Malnutrition'}, inplace=True) 
mal_min_max_yearly = mal_min_max_yearly.reset_index(drop=True)
#st.write("Min/Max malnutrition levels year-wise comparison:")
#st.dataframe(mal_min_max_yearly)


######################################################################################

#10. High CI_Width flags for monitoring(CI_width > 5)

mal_high_ci_flags = df_malnutrition[df_malnutrition['CI_Width'] > 5][['Country', 'Year', 'CI_Width']]
mal_high_ci_flags = mal_high_ci_flags.reset_index(drop=True)
mal_high_ci_flags.rename(columns={'CI_Width': 'High_CI_Width_Flag'}, inplace=True)      
#st.write("High CI_Width flags for monitoring (CI_Width > 5):")
#st.dataframe(mal_high_ci_flags)


##########################################################
#Oueries - Obesity & Malnutrition Tables
#########################################################


# 1. Obesity vs malnutrition comparison by country(any 5 countries)

ob_mal_comparison = pd.merge(df_obesity[['Country', 'Year', 'Mean_Estimate']], 
                              df_malnutrition[['Country', 'Year', 'Mean_Estimate']], 
                              on=['Country', 'Year'], 
                              suffixes=('_Obesity', '_Malnutrition'))
ob_mal_comparison = ob_mal_comparison.head(5).reset_index(drop=True)
ob_mal_comparison.rename(columns={'Mean_Estimate_Obesity': 'Obesity_Estimate', 
                                   'Mean_Estimate_Malnutrition': 'Malnutrition_Estimate'}, inplace=True)
#st.write("Obesity vs malnutrition comparison by country (any 5 countries):")
#st.dataframe(ob_mal_comparison)

#############################################################################

# 2. Gender-based disparity in both obesity and malnutrition
 


#############################################################################

# 3. Region-wise avg estimates side-by-side(Africa and America)

ob_mal_region_comparison = pd.merge(
    df_obesity.groupby('Region')['Mean_Estimate'].mean().reset_index(),
    df_malnutrition.groupby('Region')['Mean_Estimate'].mean().reset_index(),
    on='Region',
    suffixes=('_Obesity', '_Malnutrition')
)   
ob_mal_region_comparison.rename(columns={
    'Mean_Estimate_Obesity': 'Average_Obesity_Level',
    'Mean_Estimate_Malnutrition': 'Average_Malnutrition_Level'
}, inplace=True)
#st.write("Region-wise average estimates side-by-side (Africa and America):")
#st.dataframe(ob_mal_region_comparison)


#############################################################################

# 4.  Countries with obesity up & malnutrition down   

ob_mal_up_down = pd.merge(
    df_obesity.groupby('Country')['Mean_Estimate'].mean().reset_index(),
    df_malnutrition.groupby('Country')['Mean_Estimate'].mean().reset_index(),
    on='Country',
    suffixes=('_Obesity', '_Malnutrition')
)
ob_mal_up_down = ob_mal_up_down[
    (ob_mal_up_down['Mean_Estimate_Obesity'] > ob_mal_up_down['Mean_Estimate_Obesity'].mean()) &
    (ob_mal_up_down['Mean_Estimate_Malnutrition'] < ob_mal_up_down['Mean_Estimate_Malnutrition'].mean())
].reset_index(drop=True)    
ob_mal_up_down.rename(columns={
    'Mean_Estimate_Obesity': 'Average_Obesity_Level',
    'Mean_Estimate_Malnutrition': 'Average_Malnutrition_Level'
}, inplace=True)
#st.write("Countries with obesity up and malnutrition down:")
#st.dataframe(ob_mal_up_down)
                                                

#############################################################################

# 5. Age-wise trend analysis

ob_mal_age_trend = pd.merge(
    df_obesity.groupby('Age_Group')['Mean_Estimate'].mean().reset_index(),
    df_malnutrition.groupby('Age_Group')['Mean_Estimate'].mean().reset_index(),
    on='Age_Group',
    suffixes=('_Obesity', '_Malnutrition')
)                                                                   
ob_mal_age_trend.rename(columns={
    'Mean_Estimate_Obesity': 'Average_Obesity_Level',
    'Mean_Estimate_Malnutrition': 'Average_Malnutrition_Level'
}, inplace=True)
#st.write("Age-wise trend analysis:")
#st.dataframe(ob_mal_age_trend)

###############################################################################