
The following libraries may be contained as part of this distribution, according to their specified license. 
We thank the open source community for all of their contributions.

    <#list licenseMap as e>
        <#assign license = e.getKey()/>
        <#assign projects = e.getValue()/>
        <#if projects?size &gt; 0>

    ${license}:

        <#list projects as project>
        * ${project.groupId}.${project.artifactId} (${project.name}) ${project.version}
        </#list>
        </#if>
    </#list>
