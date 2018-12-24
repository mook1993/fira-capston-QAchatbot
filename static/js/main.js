function make_answer_elem(q, a) {
  var answer_pane_html = 
    '<div class="col-sm-12">'+
      '<div class="input-group mb-3 input-group-sm">'+
        '<div class="input-group-prepend">'+
          '<span class="input-group-text">Q : </span>'+
        '</div>'+
        '<input type="text" class="form-control" disabled>'+
      '</div>'+
      '<div class="input-group mb-3 input-group-sm">'+
        '<div class="input-group-prepend">'+
          '<span class="input-group-text">A : </span>'+
        '</div>'+
        '<input type="text" class="form-control" disabled>'+
      '</div>'+
    '</div>';

  elem = $.parseHTML(answer_pane_html);
  form_controls = $(elem).find('.form-control');
  $(form_controls[0]).attr('value', q)
  $(form_controls[1]).attr('value', a)

  return elem
}

const questions = {
  'vals':[
    '피보험자인 부모님이 사망하셨는데 받을 수 있는 사망보험금이 얼마나 되나요?',
    '가족이 사망하게될 시에 지급액은 얼마가 나오나요?',
    '자신이 죽게되면 지급액은 얼마인가요?',
    '자신이 죽게되면 지급액은 어떻게 되나요?',
    '본인이 사망할 시에 지급액은 얼마가 나오나요?',
    '본인이 죽게될 시에 지급액은 어떻게 되나요?',
    '가족이 죽게되면 지급액은 어떻게 되나요?'
  ],
  'random': function() {
    const index = Math.floor(Math.random() * (this.vals.length-1));
    return this.vals[index]
  }
}


$(document).ready(function(){
  var selected_tab = '#terms-1'

  $(".nav-tabs a").click(function(){
      $(this).tab('show');
      selected_tab = $(this).attr('href');
      console.log(selected_tab)
  });

  $('.btn-eraser').click(function() {
    var _selected_tab = selected_tab;
    var form_question = $(_selected_tab).find('.form-question');
    $(form_question).val('');
  });

  $('.btn-random').click(function() {
    var _selected_tab = selected_tab;
    var form_question = $(_selected_tab).find('.form-question');
    var q = questions.random()
    $(form_question).val(q);
  });

  $(".form-question").keydown(function(event) {
      if (event.keyCode == 13) {
        var _selected_tab = selected_tab;
        var btn_submit = $(_selected_tab).find('.btn-submit');
        $(btn_submit).click();
      }
  });

  $(".btn-submit").click(function(evnet) {
    var _selected_tab = selected_tab
    var question_input = $(_selected_tab).find('.question-input-pane').find('.form-control');
    if ($(question_input).val() == '') { return; }
    else {
      var _context = _selected_tab    
      var _question = $(question_input).val()
      var _data = {
        'context': _context,
        'question': _question
      }

      $.ajax('/read_qna',{
        type: 'POST',
        data: _data,
        success: function(response) {
          var answer_pane = $(_selected_tab).find('.answer-pane');
          $(answer_pane).prepend(make_answer_elem(response['question'], response['answer']))
          $(question_input).val('')
        }
      });
    }    
  });
});