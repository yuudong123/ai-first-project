const {test}=require('node:test');
const assert=require('node:assert/strict');
const {select}=require('../web/equipment-state.js');
test('설비별 센서와 이벤트 번호 유지',()=>{
  const p={event_id:99,equipment_states:[{equipment_id:'station-01',event_id:3,sensors:{PS1:10}},
    {equipment_id:'station-02',event_id:4,sensors:{PS1:20}}]};
  assert.equal(select(p,'station-01').sensors.PS1,10);
  assert.equal(select(p,'station-02').sensors.PS1,20);
  assert.equal(select(p,'station-02').event_id,4);
});
test('누락 설비를 첫 설비로 대체하지 않음',()=>{
  const p={equipment_states:[{equipment_id:'station-01',sensors:{PS1:10}}]};
  assert.deepEqual(select(p,'station-03').sensors,{});
  assert.equal(select(p,'station-03').prediction.status,'waiting');
});
test('빈 설비 배열도 공통 센서값을 복제하지 않음',()=>{
  assert.deepEqual(select({equipment_states:[],sensors:{PS1:10}},'station-01').sensors,{});
});
