"""CSRF 검증을 유지하면서 로컬 Jenkins 작업을 요청한다."""
import os
import requests


def trigger(task='retrain'):
    base = os.getenv('JENKINS_URL','http://jenkins:8080')
    session = requests.Session()
    session.auth = (os.environ['JENKINS_ADMIN_USER'],os.environ['JENKINS_ADMIN_PASSWORD'])
    crumb_response = session.get(base+'/crumbIssuer/api/json',timeout=5)
    crumb_response.raise_for_status()
    crumb = crumb_response.json()
    response = session.post(base+'/job/hydrotwin-local/buildWithParameters',
        params={'TASK':task},headers={crumb['crumbRequestField']:crumb['crumb']},timeout=5)
    response.raise_for_status()
    return response.headers.get('Location','')


if __name__ == '__main__':
    print(trigger('verify'))
