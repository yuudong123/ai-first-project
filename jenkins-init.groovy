import jenkins.model.Jenkins
import hudson.security.HudsonPrivateSecurityRealm
import hudson.security.FullControlOnceLoggedInAuthorizationStrategy
import org.jenkinsci.plugins.workflow.job.WorkflowJob
import org.jenkinsci.plugins.workflow.cps.CpsFlowDefinition
import hudson.model.ParametersDefinitionProperty
import hudson.model.ChoiceParameterDefinition

// Jenkins 이미지 시작 시 로컬 시연 계정과 파이프라인을 구성한다.
def server = Jenkins.get()
def realm = new HudsonPrivateSecurityRealm(false)
realm.createAccount(System.getenv('JENKINS_ADMIN_USER'), System.getenv('JENKINS_ADMIN_PASSWORD'))
server.setSecurityRealm(realm)
def strategy = new FullControlOnceLoggedInAuthorizationStrategy()
strategy.setAllowAnonymousRead(false)
server.setAuthorizationStrategy(strategy)
server.setNumExecutors(1)
def job = server.getItem('hydrotwin-local') ?: server.createProject(WorkflowJob, 'hydrotwin-local')
job.setDefinition(new CpsFlowDefinition(new File('/project/Jenkinsfile').text, true))
job.addProperty(new ParametersDefinitionProperty(new ChoiceParameterDefinition('TASK', ['verify','retrain'] as String[], '검증 또는 대기 중인 계절 재학습 실행')))
job.save()
server.save()
