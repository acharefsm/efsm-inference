<#-- 
	This is the template for the description section of the license header
-->

<#macro groupBy items field>
  <#if items?size == 0><#return></#if>
  <#local sorted = items?filter(item-> item[field]??)?sort_by(field)>
  <#local group = 0>
  <#local start = 0>
  <#list sorted as current>
    <#if !current?is_first && current[field] != previous[field]>
      <#nested previous[field], sorted[start ..< current?index], group>
      <#local start = current?index>
	  <#local group += 1>
    </#if>
    <#local previous = current>
  </#list>
  <#nested previous[field], sorted[start ..< sorted?size], group>
</#macro>

This file is part of ${project.name}
<#if project.properties.url??>Please visit ${project.properties.url} for further information</#if>

Authors : <#list project.developers+project.contributors as author>${author.name}<#sep>, </#list>
          <@groupBy project.developers 'organization' ; organization,developers,index><#if index gt 0>,</#if>${organization}</@groupBy>

