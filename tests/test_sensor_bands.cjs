const {test} = require('node:test');
const assert = require('node:assert/strict');
const {outside,advance} = require('../web/sensor-bands.js');
const bands = {sensors:{PS1:{lower:10,upper:20}}};
const message = (event,extra={}) => ({event_id:event,run_id:'run',segment_id:0,generated_at:new Date(event*1000).toISOString(),sensors:{PS1:21},...extra});
test('하한·상한은 포함하며 누락값은 정상으로 판정하지 않는다',()=>{
  assert.equal(outside(10,bands.sensors.PS1),false);
  assert.equal(outside(20,bands.sensors.PS1),false);
  assert.equal(outside(21,bands.sensors.PS1),true);
  assert.equal(outside(null,bands.sensors.PS1),null);
  assert.equal(outside(21,null),null);
});
test('3개 연속 이탈, 정상 복귀, 누락값 초기화',()=>{
  let state;
  for(let i=1;i<=3;i++) state=advance(state,message(i),bands,['PS1']);
  assert.equal(state.counts.PS1,3);
  assert.equal(advance(state,message(4,{sensors:{PS1:15}}),bands,['PS1']).counts.PS1,0);
  assert.equal(advance(state,message(4,{sensors:{}}),bands,['PS1']).counts.PS1,0);
});
test('재시작·구간 전환·수신 공백·중복으로 연속 횟수를 부풀리지 않는다',()=>{
  const state=advance(null,message(1),bands,['PS1']);
  for(const next of [message(1),message(3),message(2,{run_id:'new'}),message(2,{segment_id:1}),message(2,{generated_at:new Date(6000).toISOString()})]){
    assert.equal(advance(state,next,bands,['PS1']).counts.PS1,1);
  }
});
