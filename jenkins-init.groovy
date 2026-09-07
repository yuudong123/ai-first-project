import jenkins.model.Jenkins
import hudson.security.HudsonPrivateSecurityRealm
import hudson.security.FullControlOnceLoggedInAuthorizationStrategy
import org.jenkinsci.plugins.workflow.job.WorkflowJob
import org.jenkinsci.plugins.workflow.cps.CpsScmFlowDefinition
import org.jenkinsci.plugins.workflow.job.properties.PipelineTriggersJobProperty
import hudson.plugins.git.GitSCM
import hudson.plugins.git.UserRemoteConfig
import hudson.plugins.git.BranchSpec
import com.cloudbees.jenkins.GitHubPushTrigger
import com.coravy.hudson.plugins.github.GithubProjectProperty
import com.cloudbees.plugins.credentials.domains.Domain
import com.cloudbees.plugins.credentials.impl.UsernamePasswordCredentialsImpl
import com.cloudbees.plugins.credentials.CredentialsScope

// Jenkins 이미지 시작 시 로컬 시연 계정과 파이프라인을 구성한다.
def server = Jenkins.get()
def realm = new HudsonPrivateSecurityRealm(false)
realm.createAccount(System.getenv('JENKINS_ADMIN_USER'), System.getenv('JENKINS_ADMIN_PASSWORD'))
server.setSecurityRealm(realm)
def strategy = new FullControlOnceLoggedInAuthorizationStrategy()
strategy.setAllowAnonymousRead(false)
server.setAuthorizationStrategy(strategy)
server.setNumExecutors(1)

def repositoryUrl = System.getenv('GIT_REPOSITORY_URL') ?: 'https://github.com/yuudong123/ai-first-project.git'
def deployBranch = System.getenv('GIT_DEPLOY_BRANCH') ?: 'dev'
def credentialId = System.getenv('GIT_CREDENTIAL_ID') ?: 'hydrotwin-github'
def githubUser = System.getenv('GITHUB_USERNAME')
def githubToken = System.getenv('GITHUB_TOKEN')

// 비공개 저장소 토큰은 Jenkins 자격 증명 저장소에 암호화해 등록한다.
if (githubUser && githubToken) {
    def credentialsStore = com.cloudbees.plugins.credentials.SystemCredentialsProvider.getInstance().getStore()
    def existing = com.cloudbees.plugins.credentials.CredentialsProvider.lookupCredentialsInItemGroup(
        com.cloudbees.plugins.credentials.common.StandardUsernamePasswordCredentials.class,
        server,
        null,
        null
    ).find { it.id == credentialId }
    if (!existing) {
        credentialsStore.addCredentials(
            Domain.global(),
            new UsernamePasswordCredentialsImpl(
                CredentialsScope.GLOBAL,
                credentialId,
                'HydroTwin GitHub 읽기 전용 토큰',
                githubUser,
                githubToken
            )
        )
    }
}

def job = server.getItem('hydrotwin-local') ?: server.createProject(WorkflowJob, 'hydrotwin-local')
def scm = new GitSCM(
    [new UserRemoteConfig(repositoryUrl, null, null, credentialId)],
    [new BranchSpec("*/${deployBranch}")],
    false,
    [],
    null,
    null,
    []
)
def definition = new CpsScmFlowDefinition(scm, 'Jenkinsfile')
definition.setLightweight(true)
job.setDefinition(definition)
job.removeProperty(GithubProjectProperty.class)
job.addProperty(new GithubProjectProperty(repositoryUrl.replaceFirst(/\.git$/, '') + '/'))
def pushTrigger = new GitHubPushTrigger()
job.removeProperty(PipelineTriggersJobProperty.class)
job.addProperty(new PipelineTriggersJobProperty([pushTrigger]))
job.save()
pushTrigger.start(job, true)
server.save()
