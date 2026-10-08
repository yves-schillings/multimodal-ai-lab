import assert from 'node:assert/strict';
import {seedCases,trainClassifier,predict,retrieve,isLocalBackend} from '../static/core.js';
for(const identity_mode of ['simulated_loopback_only','simulated_container_network']){
 assert.equal(isLocalBackend({name:'Multimodal AI Lab',simulated_identity:true,identity_mode}),true);
}
for(const status of [undefined,{}, {name:'Other service',simulated_identity:true,identity_mode:'simulated_container_network'}, {name:'Multimodal AI Lab',simulated_identity:false,identity_mode:'simulated_loopback_only'}, {name:'Multimodal AI Lab',simulated_identity:true,identity_mode:'unknown'}]){
 assert.equal(isLocalBackend(status),false);
}
const strong=trainClassifier(),weak=trainClassifier(true);
assert.equal(strong.gate_passed,true);assert.equal(weak.gate_passed,false);
assert.equal(predict(strong.model,'Dear colleague, please reply to this email.').label,'correspondence');
const cases=seedCases();assert.equal(retrieve(cases[0],'Where was the bicycle left?').abstained,true);
cases[0].segments[0].reviewed=true;
const result=retrieve(cases[0],'Where was the bicycle left?');assert.equal(result.abstained,false);assert.ok(result.sources.every(s=>!s.case_id||s.case_id===cases[0].id));
assert.equal(retrieve(cases[0],"What colour was the suspect's jacket?").abstained,true);
console.log('Browser classification gates, reviewed-source search and missing-evidence checks passed.');
