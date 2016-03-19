/*
 * Copyright (c) 2013, Cloudera, Inc. All Rights Reserved.
 *
 * Cloudera, Inc. licenses this file to you under the Apache License,
 * Version 2.0 (the "License"). You may not use this file except in
 * compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * This software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
 * CONDITIONS OF ANY KIND, either express or implied. See the License for
 * the specific language governing permissions and limitations under the
 * License.
 */

package com.cloudera.oryx.als.computation.iterate.row;

import com.cloudera.oryx.common.collection.LongObjectMap;
import com.cloudera.oryx.common.io.DelimitedDataUtils;
import com.cloudera.oryx.common.math.SimpleVectorMath;
import com.cloudera.oryx.computation.common.fn.OryxReduceDoFn;
import org.apache.crunch.Emitter;
import org.apache.crunch.Pair;

public final class ConvergenceSampleFn extends OryxReduceDoFn<Long, float[], String> {

  private final YState yState;

  public ConvergenceSampleFn(YState yState) {
    this.yState = yState;
  }

  @Override
  public void initialize() {
    super.initialize();
    yState.initialize(getContext(), getPartition(), getNumPartitions());
  }

  @Override
  public void process(Pair<Long, float[]> input, Emitter<String> emitter) {
    // Deterministically choose one user ID for this item ID for which to compute
    // and emit its dot product. Choose the user-item pair whose hash is smallest
    long itemID = input.first();
    int itemIDHash = Long.toString(itemID).hashCode();

    int smallestUserItemHash = Integer.MAX_VALUE;
    long userIDForSmallestHash = Long.MIN_VALUE;
    float dotForSmallestHash = Float.NaN;

    for (LongObjectMap.MapEntry<float[]> entry : yState.getY().entrySet()) {
      long userID = entry.getKey();
      int hash = Long.toString(userID).hashCode() ^ itemIDHash;
      if (hash < smallestUserItemHash) {
        smallestUserItemHash = hash;
        userIDForSmallestHash = userID;
        dotForSmallestHash = (float) SimpleVectorMath.dot(input.second(), entry.getValue());
      }
    }
    if (!Float.isNaN(dotForSmallestHash)) {
      emitter.emit(DelimitedDataUtils.encode(',',
          itemID, userIDForSmallestHash, dotForSmallestHash));
    }
  }
}
